import numpy as np
from sklearn.datasets import make_blobs
# Implement K-means with oop features.

class KMeans:
    def __init__(self, X, K, max_iterations, epsilon=1e-4, repeat=100):
        pass
    
    def initialize_centroids(self):
        pass
    
    def find_closest_centroids(self):
        pass
    
    def compute_centroids(self):
        pass
    
    def compute_cost(self):
        pass
    
    def check_convergence(self):
        # Do this by using self.epsilon for checking tolerance.
        pass

    def fit(self):
        pass
    
X, _ = make_blobs(n_samples = 500, n_features=3, centers=5, random_state=1234)

kmeans = KMeans()
idx, centroids, cost_history = kmeans.fit()

# Bonus Challange:
# Try finding best K value for a given X dataset. Remember elbow tactic 
# (which is not the best, we need to engineer these features and think about them.)