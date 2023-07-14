import importlib
import numpy as np
import os
import random
import mxnet as mx
import metrics.writer as metrics_writer
import torch
from client import Client
from server import TopServer, MiddleServer
from baseline_constants import MODEL_PARAMS
from utils.args import parse_args
from utils.model_utils import read_data
from sklearn.cluster import KMeans

def main():
    args = parse_args()

    ctx = mx.gpu(args.ctx) if args.ctx >= 0 else mx.cpu()
    log_dir = os.path.join(args.log_dir, args.dataset, str(args.log_rank))
    os.makedirs(log_dir, exist_ok=True)
    log_fn = "output.%i" % args.log_rank
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")

    # Set the random seed, affects client sampling and batching
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    mx.random.seed(123 + args.seed)

    # Import the client model and server model
    client_path = "%s/client_model.py" % args.dataset
    server_path = "%s/server_model.py" % args.dataset
    if not os.path.exists(client_path) \
            or not os.path.exists(server_path):
        print("Please specify a valid dataset.",
              file=log_fp, flush=True)
        return

    client_path = "%s.client_model" % args.dataset
    server_path = "%s.server_model" % args.dataset
    mod = importlib.import_module(client_path)
    ClientModel = getattr(mod, "ClientModel")
    mod = importlib.import_module(server_path)
    ServerModel = getattr(mod, "ServerModel")

    print('load model',file=log_fp, flush=True)
    # learning rate, num_classes, and so on
    param_key = "%s.%s" % (args.dataset, args.model)
    model_params = MODEL_PARAMS[param_key]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    num_classes = model_params[1]

    # Create the shared client model
    print('create client model',file=log_fp, flush=True)
    client_model = ClientModel(
        args.seed, args.dataset, args.model, ctx, *model_params)

    # Create the shared middle server model 中间的super model
    print('create middle server model',file=log_fp, flush=True)
    middle_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    middle_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)

    # Create the top server model
    print('create top server model',file=log_fp, flush=True)
    top_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    top_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)

    # Create clients
    print('create clients',file=log_fp, flush=True)
    clients, clusters = setup_clients(client_model, args)
    print('len clusters',len(clusters),file=log_fp, flush=True)
    client_ids, client_clusters, client_num_samples = get_clients_info(clients)
    print("Total number of clients: %d" % len(clients),file=log_fp, flush=True)
    print('clusters',client_clusters,file=log_fp, flush=True)
    #计算每个cluster的平均类别分布
    cluster_dists=get_cluster_dists(clusters)
    
    
    #print("Total number of clients: %d" % len(clients))
    # Measure the global data ribution 计算P real
    global_dist, _, _ = get_clients_dist(
        clients, display=False, max_num_clients=100, metrics_dir=args.metrics_dir) #max_num_clients=20

    # Create middle servers 改为每轮r select clients 构建 supernode
#    middle_servers = setup_middle_servers(middle_server_model, middle_merged_update, clusters)
    # [middle_servers[i].brief(log_fp) for i in range(args.num_groups)]
#    print("Total number of middle servers: %d" % len(middle_servers),file=log_fp, flush=True)

    # Create the top server
    middle_servers=[]
    top_server = TopServer(top_server_model, top_merged_update, middle_servers)

    # Display initial status
#    print("--- Random Initialization ---",file=log_fp, flush=True)
    #print("--- Random Initialization ---")
    stat_writer_fn = get_stat_writer_function(
        client_ids, client_clusters, client_num_samples, args)
    print_stats(
        0, top_server, client_num_samples, stat_writer_fn,
        args.use_val_set, log_fp)
    #global_dist: List of num samples for each class.  62 class
    print("len of gd",len(global_dist))
    print("sum of gd",sum(global_dist))

    # Training simulation
    for r in range(1, args.num_rounds+1):
       print("--- Round %d of %d ---" % (r, args.num_rounds),file=log_fp, flush=True)
       supernodes,clients_per_group= top_server.select_clients(r,args.sampler,args.num_groups,args.num_class,clients, clusters, cluster_dists, global_dist,30,log_fp=None)
       s_client_ids, s_client_clusters, s_client_num_samples =get_supernodes_info(supernodes)
       middle_servers = setup_middle_servers(middle_server_model, middle_merged_update, supernodes)
       top_server.update_middleserver(middle_servers)
       # [middle_servers[i].brief(log_fp) for i in range(args.num_groups)]
       print("Total number of middle servers: %d" % len(middle_servers),file=log_fp, flush=True)
      
       # Simulate training on selected clients
       top_server.train_model(r, args.num_syncs,clients_per_group,log_fp)
       # Test model
       if r % args.eval_every == 0 or r == args.num_rounds:
           print_stats(r, top_server, s_client_num_samples, stat_writer_fn,args.use_val_set, log_fp)


    # Save the top server model
    top_server.save_model(log_dir)
    log_fp.close()

def flatten(source):
    """
    降维拼接
    """
    return torch.cat([value.flatten() for value in source.values()])
def compute_pairwise_similarities(clients):
    """
    计算客户分布的余弦相似性
    """
    return pairwise_angles([c.train_sample_dist for c in clients])
#def pairwise_angles(sources):
#    """
#    计算相似性矩阵
#    sources:所有客户分布的list
#    """
#    sources=torch.tensor(sources)
#    angles=torch.zeros([len(sources),len(sources)])
#    for i,source1 in enumerate(sources):
#        for j,source2 in enumerate(sources):
#            angles[i,j]=torch.sum(source1*source2) / (torch.norm(source1)*torch.norm(source2)+1e-12)
#    return angles.numpy()

def pairwise_angles(sources):
    """
    计算相似性矩阵
    sources:所有客户分布的list
    """
    angles=np.zeros([len(sources),len(sources)])
    for i,source1 in enumerate(sources):
        for j,source2 in enumerate(sources):
            angles[i,j]=np.sum(source1*source2) / (np.linalg.norm(x=source1,ord=2)*np.linalg.norm(x=source2,ord=2)+1e-12)
    return angles

def cluster_clients(k,S,clients):
    """
    将客户分成k个聚类
    """
    clustering=KMeans(n_clusters=k,n_init='auto',random_state=0).fit(S)
    #print('clustering',clustering)
    c_ids=[]
    for i in range(0,k):
        c_ids.append(np.argwhere(clustering.labels_ == i).flatten())
    clusters=[[]for i in range(0,k)]
    #print('c_ids',c_ids)
    for i in range(0,k):
       #print('c_ids[i]',c_ids[i])
       i_ids=c_ids[i]
       for c_id in i_ids:
           clusters[i].append((clients[c_id]))
    return clusters
    

# def create_clients(users, groups, train_data, test_data, model, args):
#     # Randomly assign a group to each client, if groups are not given    若没有组，随机给客户分组
#     random.seed(args.seed)
#     if len(groups) == 0:
#         groups = [random.randint(0, args.num_groups - 1)
#                   for _ in users]
#
#     # Instantiate clients
#     clients = [Client(args.seed, u, g, train_data[u],
#                       test_data[u], model, args.batch_size)
#                for u, g in zip(users, groups)]
#
#     return clients

def create_clients(users, clusters, train_data, test_data, model, args):
    # Randomly assign a group to each client, if groups are not given    若没有组，随机给客户分组
    random.seed(args.seed)
    if len(clusters) == 0: #先随机分cluster
        clusters = [random.randint(0, args.num_clusters - 1)
                  for _ in users]

    # Instantiate clients
    clients = [Client(args.seed, u, g, train_data[u],
                      test_data[u], model, args.batch_size)
               for u, g in zip(users, clusters)]

    return clients

# def group_clients(clients, num_groups):
#     """Collect clients of each group into a list.
#     Args:
#         clients: List of all client objects.
#         num_groups: Number of groups.
#     Returns:
#         groups: List of clients in each group.
#     """
#     groups = [[] for _ in range(num_groups)]
#     for c in clients:
#         groups[c.group].append(c)
#     return groups


# def setup_clients(model, args):
#     """Load train, test data and instantiate clients.
#     Args:
#         model: The shared ClientModel object for all clients.
#         args: Args entered from the command.
#     Returns:
#         clients: List of all client objects.
#         groups: List of clients in each group.
#     """
#     eval_set = "test" if not args.use_val_set else "val"
#     train_data_dir = os.path.join("data", args.dataset, "data", "train")
#     test_data_dir = os.path.join("data", args.dataset, "data", eval_set)
#
#     data = read_data(train_data_dir, test_data_dir)
#     users, groups, train_data, test_data = data
#
#
#     clients = create_clients(
#         users, groups, train_data, test_data, model, args)
#
#     groups = group_clients(clients, args.num_groups)
#
#     return clients, groups

def setup_clients(model, args):
    """Load train, test data and instantiate clients.
    Args:
        model: The shared ClientModel object for all clients.
        args: Args entered from the command.
    Returns:
        clients: List of all client objects.
        clusters: List of clients in each group.
    """
    eval_set = "test" if not args.use_val_set else "val"
    train_data_dir = os.path.join("data", args.dataset, "data", "train")
    test_data_dir = os.path.join("data", args.dataset, "data", eval_set)

    data = read_data(train_data_dir, test_data_dir)
    users, clusters, train_data, test_data = data
    print('before create_clients')
    clients = create_clients(
        users, clusters, train_data, test_data, model, args)
    print('after create_clients')
    print('len of clients',len(clients))
    #print('clients',clients)
    sim=compute_pairwise_similarities(clients)
    clusters=cluster_clients(args.num_clusters,sim,clients)
    #更新客户的cluster_id
    cluster_id=0
    for cluster in clusters:
        for c in cluster:
            c.set_cluster(cluster_id)
        cluster_id=cluster_id+1
    return clients, clusters

def get_clients_info(clients):
    """Returns the ids, groups and num_samples for the given clients.
    Args:
        clients: List of Client objects.
    Returns:
        ids: List of client_ids for the given clients.
        clusters: Map of {client_id: cluster_id} for the given clients.
        num_samples: Map of {client_id: num_samples} for the given
            clients.
    """
    ids = [c.id for c in clients]
    clusters = {c.id: c.cluster for c in clients}
    num_samples = {c.id: c.num_samples for c in clients}
    print('get_clients_info end')
    return ids, clusters, num_samples
    
def get_supernodes_info(supernodes):
    """Returns the ids, groups and num_samples for the given supernodes.
    Args:
        clients: List of Client objects.
    Returns:
        ids: List of client_ids for the given clients.
        clusters: Map of {client_id: cluster_id} for the given clients.
        num_samples: Map of {client_id: num_samples} for the given
            clients.
    """
    ids = [c.id  for supernode in supernodes for c in supernode ]
    clusters = {c.id: c.cluster for supernode in supernodes for c in supernode }
    num_samples = {c.id: c.num_samples for supernode in supernodes for c in supernode}
    print('get_supernodes_info end')
    return ids, clusters, num_samples

    
def get_cluster_dists(clusters):
    """
    获取每个cluster的类别分布
    N: cluster num
    K: class num
    return  N*K  matrix
    """
    cluster_dists=[]
    for cluster in clusters:
    	cluster_train_dist = sum([c.train_sample_dist for c in cluster])
    	cluster_test_dist = sum([c.test_sample_dist for c in cluster])
    	cluster_dist = cluster_train_dist + cluster_test_dist
    	cluster_dists.append(cluster_dist.tolist())
    	
    return cluster_dists

def get_clients_dist(
        clients, display=False, max_num_clients=100, metrics_dir="metrics"):
    """Return the global data distribution of all clients.
    Args:
        clients: List of Client objects.
        display: Visualize data distribution when set  True.
        max_num_clients: Maximum number of clients to plot.
        metrics_dir: Directory to save metrics files.
    Returns:
        global_dist: List of num samples for each class.
        global_train_dist: List of num samples for each class in train set.
        global_test_dist: List of num samples for each class in test set.
    """
    global_train_dist = sum([c.train_sample_dist for c in clients])
    global_test_dist = sum([c.test_sample_dist for c in clients])
    global_dist = global_train_dist + global_test_dist

    if display:

        try:
            from metrics.visualization_utils import plot_clients_dist

            np.random.seed(0)
            rand_clients = np.random.choice(clients, max_num_clients)
            plot_clients_dist(clients=rand_clients,
                              global_dist=global_dist,
                              global_train_dist=global_train_dist,
                              global_test_dist=global_test_dist,
                              draw_mean=False,
                              metrics_dir=metrics_dir)

        except ModuleNotFoundError:
            pass

    return global_dist, global_train_dist, global_test_dist


#def setup_middle_servers(server_model, merged_update, groups):
    """Instantiates middle servers based on given ServerModel objects.
    Args:
        server_model: A shared ServerModel object to store the middle
            server model.
        merged_update: A shared ServerModel object to merge updates
            from clients.
        groups: List of clients in each group.
    Returns:
        middle_servers: List of all middle servers.
    """
#    num_groups = len(groups)
#    middle_servers = [
#        MiddleServer(g, server_model, merged_update, groups[g])
#        for g in range(num_groups)]
#    return middle_servers

def setup_middle_servers(server_model, merged_update, supernodes):
    """Instantiates middle servers based on given ServerModel objects.
    Args:
        server_model: A shared ServerModel object to store the middle
            server model.
        merged_update: A shared ServerModel object to merge updates
            from clients.
        supernodes: List of clients in each supernode.
    Returns:
        middle_servers: List of all middle servers.
    """
    num_groups = len(supernodes)
    middle_servers = [
        MiddleServer(s, server_model, merged_update, supernodes[s])
        for s in range(num_groups)]
    return middle_servers


def get_stat_writer_function(ids, groups, num_samples, args):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples,
            partition, args.metrics_dir, "{}_{}_{}".format(
                args.metrics_name, "stat", args.log_rank))

    return writer_fn


def print_stats(num_round, server, num_samples, writer, use_val_set, log_fp=None):
    train_stat_metrics = server.test_model(set_to_use="train")
    #print(train_stat_metrics)
    print_metrics(
        train_stat_metrics, num_samples, prefix="train_", log_fp=log_fp)
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.test_model(set_to_use=eval_set)
    print_metrics(
        test_stat_metrics, num_samples, prefix="{}_".format(eval_set), log_fp=log_fp)
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix="", log_fp=None):
    """Prints weighted averages of the given metrics.
    Args:
        metrics: Dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: Dict with client ids as keys. Each entry is the weight
            for that client.
        prefix: String, "train_" or "test_".
        log_fp: File pointer for logs.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    #print("ordered_weights",ordered_weights)
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        #print("ordered_metric",ordered_metric)
        print("%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g" \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)),
              file=log_fp, flush=True)


if __name__ == "__main__":
    main()
