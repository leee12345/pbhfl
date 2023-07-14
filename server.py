from abc import ABC, abstractmethod
import scipy
import numpy as np
from scipy.optimize import minimize
import math
import random

class Server(ABC):
    def __init__(self, server_model, merged_update):
        self.model = server_model
        self.merged_update = merged_update
        self.total_weight = 0

    @abstractmethod
    def train_model(self, my_round, num_syncs,log_fp):
        """Aggregate clients' models after each iteration. If
        num_syncs synchronizations are reached, middle servers'
        models are then aggregated at the top server.
        Args:
            my_round: The current training round, used for learning rate
                decay.
            num_syncs: Number of client - middle server synchronizations
                in each round before sending to the top server.
            clients_per_group: Number of clients to select in
                each synchronization.
            sampler: Sample method, could be "random", "brute",
                "probability", "bayesian", "ga" (namely genetic algorithm),
                and "gbp-cs" (namely gradient-based binary permutation
                client selection).
            batch_size: Number of samples in a batch data.
            base_dist: Real data distribution, usually global_dist.
        Returns:
            update: The trained model after num_syncs synchronizations.
        """
        return None

    def merge_updates(self, weight, update):
        """Aggregate updates based on their weights.
        Args:
            weight: Weight for this update.
            update: The trained model.
        """
        merged_update_ = list(self.merged_update.get_params())
        current_update_ = list(update)
        num_params = len(merged_update_)

        self.total_weight += weight

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (weight * current_update_[p].data()))
            #print("cur update data",current_update_[p].data())

    def update_model(self):
        """Update self.model with averaged merged update."""
        merged_update_ = list(self.merged_update.get_params())
        num_params = len(merged_update_)

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() / self.total_weight)

        self.model.set_params(self.merged_update.get_params())

        self.total_weight = 0
        self.merged_update.reset_zero()

    @abstractmethod
    def test_model(self, set_to_use):
        """Test self.model on all clients.
        Args:
            set_to_use: Dataset to test on, either "train" or "test".
        Returns:
            metrics: Dict of metrics returned by the model.
        """
        return None

    def save_model(self, log_dir):
        """Save self.model to specified directory.
        Args:
            log_dir: Directory to save model file.
        """
        self.model.save(log_dir)
  
        
        
        
        

class TopServer(Server):
    def __init__(self, server_model, merged_update, servers):
        self.middle_servers = []
        self.register_middle_servers(servers)
        super(TopServer, self).__init__(server_model, merged_update)
        
    def update_middleserver(self, servers):
        self.middle_servers = servers

    def register_middle_servers(self, servers):
        """Register middle servers.
        Args:
            servers: Middle servers to be registered.
        """
        if type(servers) == MiddleServer:
            servers = [servers]

        self.middle_servers.extend(servers)

    def train_model(self, my_round, num_syncs,clients_per_group,log_fp):
        """Call middle servers to train their models and aggregate
        their updates."""
        for s in self.middle_servers:
            s.set_model(self.model)
            update = s.train_model(my_round, num_syncs,log_fp)
            self.merge_updates(clients_per_group, update)
        self.update_model()
        

    def test_model(self, set_to_use="test"):
        """Call middle servers to test their models."""
        metrics = {}

        for middle_server in self.middle_servers:
            middle_server.set_model(self.model)
            s_metrics = middle_server.test_model(set_to_use)
            metrics.update(s_metrics)

        return metrics
    

    def get_dist_distance(self, num_class,selected_clusters, cluster_dists, base_dist, use_distance="l2"):
        """Return distance of the base distribution and the mean distribution.
        Args:
            clients: List of sampled clients.
            base_dist: Real data distribution, usually global_dist.
            use_distance: Distance metric to be used, could be:
                ["l1", "l2", "cosine", "js", "wasserstein"].
        Returns:
            distance: The distance of the base distribution and the mean
                distribution.
        """
        temp=[np.array([selected_clusters[i]* cluster_dists[i][j] for j in range(num_class)]) for i in range(len(cluster_dists))]
        #print(temp)
        c_sum_samples_ = sum(temp) # class_num 62 大小的list  每个class的sample数量
        #print("c_sum_samples_",c_sum_samples_)
        c_mean_dist_ = c_sum_samples_ / c_sum_samples_.sum()#list 每个class的sample比例
        #print(c_sum_samples_.sum())#一个值 全部sample数
        #print("c_mean_dist_",c_mean_dist_)
        
        base_dist_ = base_dist / base_dist.sum()

        distance = np.inf
        if use_distance == "l1":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=1)
        elif use_distance == "l2":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=2)
        elif use_distance == "cosine":
            # The cosine distance between vectors u and v is defined as:
            #       1 - dot(u, v) / (norm(u, ord=2) * norm(v, ord=2))
            distance = scipy.spatial.distance.cosine(c_mean_dist_, base_dist_)
        elif use_distance == "js":
            distance = scipy.spatial.distance.jensenshannon(c_mean_dist_, base_dist_)
        elif use_distance == "wasserstein":
            distance = scipy.stats.wasserstein_distance(c_mean_dist_, base_dist_)

        return distance
            
        
    def select_clients(self, my_round, sampler, num_groups, num_class,clients, clusters, cluster_dists, base_dist,ep,log_fp=None):# rand_per_group=2
        """
        原在middle server下的select——clients 改为在top server下
        返回 clients_per_group: 每个supernpde的客户数量
        """
                            
        if sampler == "lagrangian":
            selected_clusters,clients_per_group = self.lagrangian_sampling(num_class,clusters, num_groups,cluster_dists, base_dist,ep)
        elif sampler == "random":
            selected_clusters,clients_per_group = self.random_sampling(num_class,clusters, num_groups,cluster_dists, base_dist,ep)
            
        np.random.seed(my_round)
	#selected_clients = np.random.choice(clients, num_clients, replace=False).tolist()
        # Measure the distance of base distribution and mean distribution
        distance = self.get_dist_distance(num_class,selected_clusters,cluster_dists, base_dist)
        print("Dist Distance:", distance, file=log_fp,flush=True)
        # random select clients from clusters
        supernodes=[[]for i in range(num_groups)]
        col=0
        for i in range(len(clusters)):
            rand_clients_ = np.random.choice(clusters[i], num_groups*selected_clusters[i], replace=False).tolist()
            for j in range(num_groups):
                supernodes[j].extend(rand_clients_[col:col+selected_clusters[i]])
                col+=selected_clusters[i]
        #print(supernodes)
        return supernodes,clients_per_group



    def lagrangian_sampling(self, num_class,clusters, num_groups ,cluster_dists, base_dist,ep):
        """
         fx
        """
        def get_map(L,B):
            the_map=[]
            for i in range(len(clusters)):
                n=1
                l=L[i]
                b=B[i]
                for j in range(l-1):
                    the_map.append(n)
                    b-=n
                    n*=2
                the_map.append(b)
            return the_map
            
        def get_A(M,L,num_class,cluster_dists,the_map):
            A=[[0 for j in range(num_class)] for i in range(M)]
            cur=0
            for i in range(len(cluster_dists)):
                l=L[i]
                for k in range(l):
                    for j in range(num_class):
                        A[cur][j]=cluster_dists[i][j]*the_map[cur]
                    cur+=1
            return A
            
        def get_selected_clusters(clusters,x_idx,L,the_map):
            selected_clusters=[0 for i in range(len(clusters))]
            cur=0
            for i in range(len(clusters)):
                l=L[i]
                for k in range(l):
                    selected_clusters[i]+=int(x_idx[cur]*the_map[cur])
                    cur+=1
            return selected_clusters,sum(selected_clusters)
            
        def get_bin(x):
            return bin(x).replace('0b','') 
            
        def get_binary(x):
            return x*(1-x)

        def penalty_function(x, l, t):
           tmp = 0
           for i in range(0,len(x)):
               tmp += l[i]*get_binary(x[i])+t/2*get_binary(x[i])**2
           return tmp

        def f(x,C,q):
           return np.dot(np.dot(np.array(x).T,C),np.array(x))+np.dot(np.array(q),np.array(x))

        def update_lmbda(lmbda, x, t):
           for i in range(len(lmbda)):
               #print("lmbda",lmbda)
               #print(t*get_binary(x[i]))
               #lmbda += t*get_binary(x[i])
               lmbda[i] += t*get_binary(x[i])

        def augmented_lagrangian(x,lmbda,t,C,b):
           return f(x,C,b)+penalty_function(x,lmbda,t)

        def augmented_largrangian_optimize(x0, l, t):
           """
           x0:初值  设置为一堆1
           l:lambda
           t:惩罚系数
           """
           x = x0
           lmbda = l
           for i in range(ep):
               res = minimize(lambda x: augmented_lagrangian(x,lmbda,t,Q,p),x,method='L-BFGS-B').x
               t = t*1.01
               update_lmbda(lmbda,x, t)
               x = res
           return x
        

        B=[math.floor(len(cluster)/num_groups) for cluster in clusters]
        L=[math.ceil(math.log(b,2)) for b in B]
        M=sum(L)
        the_map=get_map(L,B)# len:M  map of value
        A=get_A(M,L,num_class,cluster_dists,the_map)
        Q = np.dot(A, np.transpose(A))
        p = -2*np.dot(base_dist, np.transpose(A))
        # C = QQ^T-2*diag(p)
        # C = matrix(np.dot(np.transpose(A), A)-2*np.diag(p))
        res = augmented_largrangian_optimize([1.0]*M, [1.0]*M, 1.0)# 每列对应一个值，接近1的代表选
        #print("res",res)
        # 存疑，不知道是不是应该这样写，对client不太清楚。如果是不限制client的数量的话，那么就是选择res中大于0.5的client。
        #x_idx = np.argsort(res)[-num_clients:] 
        x_idx=[abs(round(x,0)) for x in res]
        #print("x_idx",x_idx)
        selected_clusters,clients_per_group=get_selected_clusters(clusters,x_idx,L,the_map)
        return selected_clusters,clients_per_group
        
    def random_sampling(self, num_class,clusters, num_groups ,cluster_dists, base_dist,ep):
        np.random.seed(ep)
        B=[math.floor(len(cluster)/num_groups) for cluster in clusters]
        selected_clusters=[]
        for i in range(len(clusters)):
            selected_clusters.append(random.randint(0,B[i]))
        clients_per_group=sum(selected_clusters)
        return selected_clusters,clients_per_group


class MiddleServer(Server):
    def __init__(self, server_id, server_model, merged_update, clients_in_group):
        self.server_id = server_id
        self.clients = []
        self.register_clients(clients_in_group)
        super(MiddleServer, self).__init__(server_model, merged_update)

    def register_clients(self, clients):
        """Register clients of this middle server.
        Args:
            clients: Clients to be registered.
        """
        if type(clients) is not list:
            clients = [clients]

        self.clients.extend(clients)



    

    def train_model(self, my_round, num_syncs,log_fp):
        """def train_model(self, my_round, num_syncs, clients_per_group,sampler, batch_size, base_dist,log_fp):"""
        """Train self.model for num_syncs synchronizations."""
        print('tain model start')
        for _ in range(num_syncs):
            print('syncs',_)
            # Select clients for current synchronization 
            # middle server每次训练都从中选择客户参与训练
            #selected_clients = self.select_clients(my_round, clients_per_group, sampler, batch_size, base_dist,log_fp)

            # Train on selected clients for one iteration
            for c in self.clients:
                c.set_model(self.model)
                comp, num_samples, update = c.train(my_round)
                self.merge_updates(num_samples, update)

            # Update model of middle server
            self.update_model()

        update = self.model.get_params()
        print('tain model end')
        return update

    def test_model(self, set_to_use="test"):
        """Test self.model on online clients."""
        s_metrics = {}

        for client in self.online(self.clients):
            client.set_model(self.model)
            c_metrics = client.test(set_to_use)
            s_metrics[client.id] = c_metrics

        return s_metrics

    def set_model(self, model):
        """Set the model data to specified model.
        Args:
            model: The specified model.
        """
        self.model.set_params(model.get_params())

    def online(self, clients):
        """Return clients that are online.
        Args:
            clients: List of all clients registered at this
                middle server.
        Returns:
            online_clients: List of all online clients.
        """
        online_clients = clients
        assert len(online_clients) != 0, "No client available."
        return online_clients

    @property
    def num_clients(self):
        """Return the number of all clients registered at this
        middle server."""
        if not hasattr(self, "_num_clients"):
            self._num_clients = len(self.clients)

        return self._num_clients

    @property
    def num_samples(self):
        """Return the total number of samples for self.clients."""
        if not hasattr(self, "_num_samples"):
            self._num_samples = sum([c.num_samples for c in self.clients])

        return self._num_samples

    @property
    def num_train_samples(self):
        """Return the total number of train samples for
        self.clients."""
        if not hasattr(self, "_num_train_samples"):
            self._num_train_samples = sum([c.num_train_samples
                                           for c in self.clients])

        return self._num_train_samples

    @property
    def num_test_samples(self):
        """Return the total number of test samples for
        self.clients."""
        if not hasattr(self, "_num_test_samples"):
            self._num_test_samples = sum([c.num_test_samples
                                          for c in self.clients])

        return self._num_test_samples

    @property
    def train_sample_dist(self):
        """Return the distribution of train data for
        self.clients."""
        if not hasattr(self, "_train_sample_dist"):
            self._train_sample_dist = sum([c.train_sample_dist
                                           for c in self.clients])

        return self._train_sample_dist

    @property
    def test_sample_dist(self):
        """Return the distribution of test data for
        self.clients."""
        if not hasattr(self, "_test_sample_dist"):
            self._test_sample_dist = sum([c.test_sample_dist
                                          for c in self.clients])

        return self._test_sample_dist

    @property
    def sample_dist(self):
        """Return the distribution of overall data for
        self.clients."""
        if not hasattr(self, "_sample_dist"):
            self._sample_dist = self.train_sample_dist + self.test_sample_dist

        return self._sample_dist

    def brief(self, log_fp):
        """Briefly summarize the statistics of this middle server"""
        print("[Group %i] Number of clients: %i, number of samples: %i, "
              "number of train samples: %s, number of test samples: %i, "
              % (self.server_id, self.num_clients, self.num_samples,
                 self.num_train_samples, self.num_test_samples),
              file=log_fp, flush=True, end="\n")
        print("sample distribution:", list(self.sample_dist.astype("int64")),file=log_fp, flush=True)

