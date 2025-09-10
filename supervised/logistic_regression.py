import numpy as np
import copy
from math import ceil
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer # Dataset to train & try out our model


class LogisticRegression:
    def __init__(self, max_iterations=1000, alpha=1.0e-1, lambda_=0.01, epsilon=1.0e-10, w_init=np.array([]), b_init=1, scale_features=True, rand_seed=1234):
        self.iterations = 0
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.w = w_init
        self.b = b_init
        self.scale_features = scale_features
        np.random.seed = rand_seed
        
        
    def zscore_normalization(self, X):
        # Find the mean of each column
        mu = np.mean(X, axis=0)
        # Find the standard deviation of each column
        sigma = np.std(X, axis=0)
        # Element-wise scaling by substracting mu and dividing the result with standard deviation for that column.
        X_normalized = (X - mu) / sigma

        return X_normalized
    
    
    def get_model(self):
        return lambda X : 1 / (1 + np.exp( -(np.dot(self.w, X) + self.b) ))
    
    
    def predict(self, X, scale=False):
        model = self.get_model()
        
        if scale:
            X = self.zscore_normalization(X)
        
        return model(X)
    
    
    def compute_cost(self, x_train, y_train):
        m, n = x_train.shape
        
        loss_cost = 0
        for i in range(m):
            f = self.predict(x_train[i])
            loss = -y_train[i] * np.log(f) - (1 - y_train[i]) * np.log(1 - f)
            loss_cost += loss
        loss_cost = loss_cost / m
        
        # Regularization
        reg_cost = 0
        for j in range(n):
            reg_cost += self.w[j] ** 2
        reg_cost = reg_cost * self.lambda_ / m
        
        total_cost = loss_cost + reg_cost
        return total_cost
    
            
    def compute_gradient(self, x_train, y_train):
        m, n = x_train.shape
        
        gradient_w = np.zeros((n,))
        gradient_b = 0
        
        for i in range(m):
            err = self.predict(x_train[i]) - y_train[i]
            
            for j in range(n):
                gradient_w[j] += err * x_train[i, j] 
            gradient_b += err
            
        gradient_w = gradient_w / m 
        gradient_b = gradient_b / m
        
        return gradient_w, gradient_b
    
        
    def gradient_descent(self, x_train, y_train):        
        m, n = x_train.shape
        
        for i in range(self.max_iterations):
            gradient_w, gradient_b = self.compute_gradient(x_train, y_train)
            cost = self.compute_cost(x_train, y_train)
            
            self.iterations += 1 # increment iteration counter.
            
            for j in range(n):
                reg_term = self.lambda_ * self.w[j] / m
                self.w[j] = self.w[j] - self.alpha * (gradient_w[j] + reg_term)

            self.b = self.b - self.alpha * gradient_b
            
            if i % (ceil(self.max_iterations) / 10) == 0:
                # Print out the cost, w, b 10 times throughout the gradient descent so that we can see our progress.
                print(f"#{i} - cost: {cost}, w: {self.w}, b: {self.b}, gradient_w: {gradient_w}, gradient_b: {gradient_b}")
            
            # If w or b stops incrementing or increments as smaller than epsilon, then return because our gradient descent converged. 
            if abs(max(gradient_w)) < self.epsilon and abs(gradient_b) < self.epsilon:
                break
    
    
    def fit(self, x_train, y_train):
        X = copy.deepcopy(x_train)
        
        if self.scale_features:
            X = self.zscore_normalization(X)
        
        m, n = X.shape
        if n != self.w.shape[0]:
            self.w = np.random.random((n,))
            
        self.gradient_descent(X, y_train)
        
        print(f"Gradient descent completed in {self.iterations} iterations.")
        print(f"w, b found by gradient descent: {self.w}, {self.b}")
        
    
    def plot_model(self, x_train, y_train):
        model = self.get_model()
        y_predictions = np.array([model(i) for i in self.zscore_normalization(x_train)]) # This line creates a numpy array with all prediction values.

        plt.scatter(range(len(y_train)), y_train, color='b', label='Actual')
        plt.scatter(range(len(y_predictions)), y_predictions, color='r', marker='x', label='Predicted')
        
        plt.title("Actual vs Predicted values")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
# USAGE #
        
# Creating Logistic Regression instance
logistic = LogisticRegression(max_iterations=1000)

# Load dataset - sklearn breast cancer dataset.
data = load_breast_cancer()
x_train, y_train = data.data, data.target

# Train Logistic Regression Model. Note that this takes a few minutes.
logistic.fit(x_train, y_train)

# Lets try predicting y_train[0] value with x_train[0] 
y_prediction, y_actual = logistic.predict(x_train, scale=True), y_train[0]

print("| y_prediction - y_actual |")
print(y_prediction, " -", y_actual)

# Plot the trained model to see how it performs
logistic.plot_model(x_train, y_train)
