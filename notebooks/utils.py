import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_classification

# A place to place activation, loss, normalizer and plot functions.

class Activation:
    def __init__(self, l_activation, l_gradient):
        self.l_activation = l_activation
        self.l_gradient = l_gradient
        
    def __call__(self, Z:np.ndarray):
        self.Z = Z
        self.A = self.l_activation(Z)
        return self.A
    
    def gradient(self):
        return self.l_gradient(self)


class Activations:
    @staticmethod
    def Sigmoid():
        return Activation(lambda Z: 1 / (1 + np.exp(-Z)), lambda self: self.A * (1 - self.A))
    
    @staticmethod
    def ReLU():
        return Activation(lambda Z: np.maximum(0, Z), lambda self: (self.Z > 0).astype(float))
    
    @staticmethod
    def Tanh():
        return Activation(lambda Z: np.tanh(Z), lambda self: 1 - self.A ** 2)
    
    @staticmethod
    def Softmax():
        """Made only for using with CategoricalCrossentropy Loss!"""
        def softmax(Z):
            shift = Z - np.max(Z, axis=1, keepdims=True)
            exp = np.exp(shift)
            return exp / np.sum(exp, axis=1, keepdims=True)
            
        return Activation(softmax, lambda self: self.A * (1 - self.A))


class Loss:
    def __init__(self, l_loss, l_gradient, clip_Yhat=False):
        self.l_loss, self.l_gradient, self.clip_Yhat = l_loss, l_gradient, clip_Yhat

    def __call__(self, Y_hat:np.ndarray, Y:np.ndarray):
        self.Y_hat, self.Y = Y_hat, Y
        if self.clip_Yhat: self.Y_hat = np.clip(self.Y_hat, 1e-15, 1 - 1e-15)
        return self.l_loss(self.Y_hat, self.Y)

    def gradient(self):
        return self.l_gradient(self)
        
        
class Losses:
    @staticmethod
    def MeanSquaredError():
        l_loss = lambda Y_hat, Y: np.mean((Y_hat - Y) ** 2)
        l_gradient = lambda self: 2 * (self.Y_hat - self.Y) / self.Y.shape[0]
        return Loss(l_loss, l_gradient)
    
    @staticmethod
    def BinaryCrossEntropy():
        l_loss = lambda Y_hat, Y: -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        l_gradient = lambda self: (self.Y_hat - self.Y) / (self.Y_hat * (1 - self.Y_hat) * self.Y.shape[0])
        return Loss(l_loss, l_gradient, clip_Yhat=True)
    
    @staticmethod
    def CategoricalCrossEntropy():
        l_loss = lambda Y_hat, Y: -np.mean(np.sum(Y * np.log(Y_hat), axis=1))
        l_gradient = lambda self: -(self.Y / self.Y_hat) / self.Y.shape[0]
        return Loss(l_loss, l_gradient, clip_Yhat=True)
    
    
class ZNormalizer: # ZScore Normalizer
    def __init__(self, ):
        self.mu = None
        self.sigma = None
    
    def adjust(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
    
    def norm(self, X):
        if self.mu == None and self.sigma == None:
            self.adjust(X)
            
        X_normalized = (X - self.mu) / self.sigma
        return X_normalized
    
# TODO: Implement
class Plotting:
    pass

class Data:
    @staticmethod
    def yield_batches(X, Y, batch_size=32, shuffle=True):
        m = X.shape[0]
        indices = np.arange(m)
        
        if shuffle: np.random.shuffle(indices)
        for i in range(0, m, batch_size):
            j = i + batch_size
            batch_indices = indices[i:j]
            
            yield X[batch_indices], Y[batch_indices]

    @staticmethod
    def create_batches(X, Y, batch_size=32, shuffle=True):
        m, n = X.shape
        indices = np.arange(m)
        X_Batches = []
        Y_Batches = []
        if shuffle: np.random.shuffle(indices)
        
        for i in range(0, m, batch_size):
            j = i + batch_size
            batch_indices = indices[i:j]
            X_Batches.append(X[batch_indices])
            Y_Batches.append(Y[batch_indices])
            
        return X_Batches, Y_Batches

"""
X, y = make_classification(
    n_samples=500,        # total samples
    n_features=2,         # number of features for easy plotting
    n_informative=2,      # informative features
    n_redundant=0,        # redundant features
    n_clusters_per_class=1,
    n_classes=2,          # binary classification
    random_state=1234
)

X_batch, Y_batch = Data.create_batches(X, y)
Y_hat_all = np.vstack(Y_batch)  # stack batches vertically
print(Y_hat_all)

MSE = Losses.MeanSquaredError()
BCE = Losses.BinaryCrossEntropy()
CCE = Losses.CategoricalCrossEntropy()

#loss = CCE(y, y)
#print(loss)
"""