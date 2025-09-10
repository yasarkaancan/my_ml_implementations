import numpy as np

# Gaussian Distribution (Maximum likelihood estimate for mu and sigma)
#p(x) = 1 / (   ((2 * pi) ** 0.5 * sigma) * ( e ** (-(x - mu) ** 2) / (2 * sigma ** 2))  )

# sigma = standard deviation && sigma ** 2 = variance

# Vectorized (multiple featured) P(X) = P(X1) * P(X2) * ... * P(Xn)

def gaussian_distribution(X):
    def p(x, sigma, mu):
        return 1 / (((2 * np.pi) ** 0.5 * sigma) * (np.exp(-(x - mu) ** 2) / (2 * sigma ** 2)))
    mu = np.mean(X) # for a single feature!
    
# Real number evaluation

# Lets say that we have a dataset with 10000 good examples, 20 anomalies.
# In this scenario, we would do this:
# Training Set  : 6000 good
# CV Set        : 2000 good, 10 anomaly
# Test Set      : 2000 good, 10 anomaly

# We would use CV & Test sets to tune epsilon value which we use to determine if the output probability is normal or not. ( p(x) < epsilon | for x = anomaly)

# When choosing features to use, best practice is plotting them with plt.hist() and see if they're bell shaped.
# If not, then we would transform it to the bell shape by trying np.log(x + c) or x ** 0.5 
