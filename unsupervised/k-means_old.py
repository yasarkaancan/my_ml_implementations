# elbow method:

# run from k = range(1, 9) and select the "elbow" point where the cost really does not decrease as much as before.

import numpy as np
from sklearn.datasets import make_blobs

class Kmeans:
    def __init__(self, X, K, max_iterations, repeat=100, run_log=False, converge_log=True, centroid_log=False):
        self.X = X
        self.K = K
        self.max_iterations = max_iterations
        self.repeat = repeat
        self.cost_history, self.idx_history, self.centroid_history = [], [], []
        self.run_log, self.converge_log, self.centroid_log = run_log, converge_log, centroid_log

    @staticmethod
    def find_closest_centroids(X, centroids):
        m, n = X.shape
        idx = np.zeros(m, dtype=int)
        
        for i in range(m):
            # This is the L2 norm of all centroids and np.argmin() returns the index of closest centroid to the point X[i]
            idx[i] = np.argmin(np.linalg.norm(centroids - X[i], axis=1))
            
        return idx


    

    @staticmethod
    def compute_centroids(X, idx, K, centroid):
        m, n = X.shape
        centroids = np.zeros((K, n))
        
        for k in range(K):
            points = X[idx == k]
            if len(points) > 1:
                centroids[k] = np.mean(points, axis=0)
            else:
                centroids[k] = X[np.random.choice(m)]
        return centroids


    @staticmethod
    def compute_cost(X, idx, centroids):
        K, cost = len(centroids), 0
        
        for k in range(K):
            cost += np.mean(np.linalg.norm(centroids[k] - X[idx == k], axis=1), axis=0)
        return cost
    
    
    @staticmethod
    def check_convergence(cost_history, i):
        if i > 0 and cost_history[i] == cost_history[i - 1]:
            return True
        return False 
    
    
    @staticmethod
    def choose_best_clustering(cost_history, centroid_history, idx_history, K):
        cost_history = np.array(cost_history)
        best_repeat = np.argmin(cost_history)
        
        return idx_history[best_repeat], centroid_history[best_repeat][-1], best_repeat
    
    
    def run(self):
        for repeat in range(self.repeat):
            i, idx, centroids, cost_history, converged = 0, np.zeros(self.X.shape[0], dtype=int), X[np.random.permutation(self.X.shape[0])][:self.K] , np.zeros(self.max_iterations), False
            
            while not converged and i < self.max_iterations:
                idx = self.find_closest_centroids(self.X, centroids)
                centroids = self.compute_centroids(self.X, idx, self.K, centroids)
                
                cost = self.compute_cost(self.X, idx, centroids)
                
                cost_history[i] = cost
                converged = self.check_convergence(cost_history, i)
                
                if self.run_log:
                    print(f"#{repeat} - K-Means run #{i}/{self.max_iterations} | Cost: {cost} ")
                    if self.centroid_log: print(f"#{repeat} - Centroids state run #{i}: {centroids}")
                i += 1
            
            self.centroid_history.append(centroids)
            self.cost_history.append(cost)
            self.idx_history.append(idx)
            
            if self.converge_log:
                print(f"#{repeat} - Converged at run #{i-1}! | Cost: {cost}")
                if self.centroid_log: print(f"Centroid states under convergence: {centroids}")
                
                print("------------------")
                
        idx, centroids, best_repeat = self.choose_best_clustering(self.cost_history, self.centroid_history, self.idx_history, self.K)
        
        print(f"Best repeat was #{best_repeat} | Cost: {self.cost_history[best_repeat]}")
        return idx, centroids
        
        
    # Decide on K value
            
X,y = make_blobs(n_samples = 500,n_features = 3,centers = 5,random_state = 1234)

kmeans = Kmeans(X, K=5, max_iterations=100, repeat=1000, centroid_log=False, run_log=False)

# Returns the best choice of idx (this is a list of centroid index values, that is assigned to X data at that index; i.e. if idx[0] = 1, then X[0] is assigned to cluster 1 and so on.) and centroids 
idx, centroids = kmeans.run() 

print(centroids)