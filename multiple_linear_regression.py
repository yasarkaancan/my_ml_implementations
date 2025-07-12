"""
Hi! This is a simple, multiple variable linear regression code for predicting values with multiple features.

Class Features:
    - Feature Scaling with Z-Score scaling.
    - Model Training with Gradient Descent
    - Model Generation with the found coefficients w and b (Found by Gradient Descent).
    - Checking for convergence with epsilon value and gradient value.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import copy

class LinearRegression:
    def __init__(self, x_train, y_train, max_iterations=10000, alpha=0.01, epsilon=1.0e-10, w_init=None, b_init=100):
        
        # We need to scale these X features so that our model is trained better.
        scaled_x_train = self.zscore_normalization(x_train)

        self.x_train = scaled_x_train # x training values
        self.y_train = y_train # y training values
        self.iterations = 0 # Iteration count
        self.max_iterations = max_iterations # Number of iterations for gradient descent algorithm
        self.epsilon = epsilon # Epsilon value for our gradient descent to stop if gradient is lower than epsilon.
        self.alpha = alpha # Learning Rate
        
        # Initial values for w, b
        if w_init == None:
            self.w_init = np.zeros_like(scaled_x_train[0])
        
        self.b_init = b_init
        
        # Run Gradient Descent
        self.w, self.b = self.gradient_descent(self.max_iterations, self.alpha) 
    
    
    def zscore_normalization(self, X):
        # Find the mean of each column
        mu = np.mean(X, axis=0)
        # Find the standard deviation of each column
        sigma = np.std(X, axis=0)
        # Element-wise scaling by substracting mu and dividing the result with standard deviation for that column.
        X_normalized = (X - mu) / sigma

        return X_normalized
    
    
    def model_generate(self, w, b):
        return lambda x: np.dot(x, w) + b
    
    
    def compute_cost(self, w, b):
        m = self.x_train.shape[0]
        model = self.model_generate(w, b)
        # Calculates the cost - how well our model fits to the actual data - (lower == better)
        cost = 0
        for i in range(m):
            cost += (model(self.x_train[i]) - self.y_train[i]) ** 2
        cost /= (2 * m)
        
        return cost
    
    
    def compute_gradient(self, w, b):
        m, n = self.x_train.shape
        
        gradient_w = np.zeros((n,))
        gradient_b = 0
        model = self.model_generate(w, b)
        for i in range(m):
            err = model(self.x_train[i]) - self.y_train[i]
            
            # Gradient of the cost function with respect to coefficients w and b
            for j in range(n):
                gradient_w[j] = gradient_w[j] + err * self.x_train[i, j]
            gradient_b += err
            
        gradient_w = gradient_w / m
        gradient_b = gradient_b / m
        
        return gradient_w, gradient_b
    
    
    def gradient_descent(self, max_iterations, alpha):
        w = copy.deepcopy(self.w_init)
        b = self.b_init
                        
        for i in range(max_iterations):
            gradient_w, gradient_b = self.compute_gradient(w, b)
            cost = self.compute_cost(w, b)
            
            w = w - alpha * gradient_w
            b = b - alpha * gradient_b
            
            self.iterations = self.iterations + 1
            
            if i % (ceil(max_iterations) / 10) == 0:
                # Print out the cost, w, b 10 times throughout the gradient descent so that we can see our progress.
                print(f"#{i} - cost: {cost}, w: {w}, b: {b}")
            
            # If w or b stops incrementing or increments as smaller than epsilon, then return because our gradient descent converged. 
            if abs(max(gradient_w)) < self.epsilon and abs(gradient_b) < self.epsilon:
                return w, b
        return w, b
    
    
    def get_model(self):
        return self.model_generate(self.w, self.b)


    def plot_model(self):
        model = self.get_model()
        y_predictions = np.array([model(i) for i in linear_reg.zscore_normalization(x_train)]) # This line creates a numpy array with all prediction values.

        plt.scatter(range(len(self.y_train)), self.y_train, color='b', label='Actual')
        plt.scatter(range(len(y_predictions)), y_predictions, color='r', marker='x', label='Predicted')

        plt.title("Actual vs Predicted values")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def __str__(self):
        return f"Linear Regression model found by Gradient Descent is:\nF(x) = {self.w} . X + {self.b}\nIterations Completed : {self.iterations}"

## Training & using linear regression model
    
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]) # Example training features
y_train = np.array([460, 232, 178])

linear_reg = LinearRegression(x_train, y_train, max_iterations=10000) # Gradient Descent will run and find optimal w and b

print(linear_reg) # This will run __str__() method.

F = linear_reg.get_model() # This is our model with trained coefficients.
# You can use this model to predict.
x_test = linear_reg.x_train[0] 

# !!! Notice that I am not using x_train[0] directly. 
# This is because the class variable x_train is a scaled version of the actual x_train.
# This was a note that remarks you should only predict with scaled features!

prediction = F(x_test) # Prediction Example
real = y_train[0]

print(f"Real value: {real} --- Prediction: {prediction}")

# Plotting all actual values VS predicted values.

linear_reg.plot_model()
