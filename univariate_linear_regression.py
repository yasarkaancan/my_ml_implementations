"""
Hi! This is a simple, one variable linear regression code for predicting only one feature.

Class Features:
    - Model Training with Gradient Descent
    - Model Generation with the found coefficients w and b (Found by Gradient Descent).
    - Checking for convergence with epsilon value.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

class LinearRegression:
    def __init__(self, x_train, y_train, max_iterations=10000, alpha=0.01, epsilon=0.00001, w_init=100, b_init=100):
        self.x_train = x_train # x training values
        self.y_train = y_train # y training values
        self.iterations = 0 # Iteration count
        self.max_iterations = max_iterations # Number of iterations for gradient descent algorithm
        self.epsilon = epsilon # Epsilon value for our gradient descent to stop if gradient is lower than epsilon.
        self.alpha = alpha # Learning Rate
        
        # Initial values for w, b
        self.w_init = w_init
        self.b_init = b_init
        
        # Run Gradient Descent
        self.w, self.b = self.gradient_descent(self.max_iterations, self.alpha) 
        
        
    def model_generate(self, w, b):
        return lambda x: w * x + b # Returns the function F(x) with coefficients w and b
    
    
    def compute_cost(self, w, b):
        m = len(self.x_train)
        model = self.model_generate(w, b)
        # Calculates the cost - how well our model fits to the actual data - (lower == better)
        cost = 0
        for i in range(m):
            cost += (model(self.x_train[i]) - self.y_train[i]) ** 2
        cost /= (2 * m)
        
        return cost
    
    
    def compute_gradient(self, w, b):
        m = len(self.x_train)
        gradient_w = 0
        gradient_b = 0
        model = self.model_generate(w, b)
        
        for i in range(m):
            x_prediction = model(self.x_train[i])
            
            # Gradient of the cost function with respect to coefficients w and b
            gradient_w += (x_prediction - self.y_train[i]) * x_train[i]
            gradient_b += (x_prediction - self.y_train[i])
        
        gradient_w /= m
        gradient_b /= m
        
        return gradient_w, gradient_b
    
    
    def gradient_descent(self, max_iterations, alpha):    
        w = self.w_init
        b = self.b_init
                        
        for i in range(max_iterations):
            gradient_w, gradient_b = self.compute_gradient(w, b)
            
            w = w - alpha * gradient_w
            b = b - alpha * gradient_b
            
            self.iterations = self.iterations + 1
            if i % (ceil(max_iterations) / 10) == 0:
                # Print out the cost, w, b 10 times throughout the gradient descent so that we can see our progress.
                print(f"#{i} - cost: {self.compute_cost(w, b)}, w: {w}, b: {b}")
            
            # If w and b does not change much and their change is lower than epsilon, return early.
            if abs(gradient_w) < self.epsilon and abs(gradient_b) < self.epsilon:
                print(f"Gradient Descent Completed in #{self.iterations} iterations.")
                return w, b
        return w, b
    
    
    def get_model(self):
        return self.model_generate(self.w, self.b)
    
    
    def __str__(self):
        return f"Linear Regression model found by Gradient Descent is:\nf(x) = {self.w}*x + {self.b}\nIterations Completed : {self.iterations}"
        

## Training & using linear regression model
    
x_train = np.array([1., 2., 3., 4., 5.])
y_train = np.array([300, 500, 700, 800, 801])

linear_reg = LinearRegression(x_train, y_train, max_iterations=10000) # Gradient Descent will run and find optimal w and b
print(linear_reg) # This will run __str__() method.

F = linear_reg.get_model() # Prediction Model

# Lets plot and see how well our prediction model performs compared to real data.
y_predictions = np.array([F(i) for i in x_train]) # This line creates a numpy array with all prediction values.


plt.scatter(x_train, y_train, color='r', label='Actual y values', alpha=0.6)
plt.plot(x_train, y_predictions, color='b', label='Predictions from F(x) ', linewidth=2)

plt.title('Actual vs Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
    
# Thanks for coming this far!
