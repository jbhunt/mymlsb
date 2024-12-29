import numpy as np

class SimpleLinearRegression():
    """
    """

    def __init__(self):
        """
        """

        self.beta0 = np.nan
        self.beta1 = np.nan

        return
    
    def fit(self, X, y):
        """
        """

        self.beta1 = (np.sum([(xi - np.mean(X)) * (yi - np.mean(y)) for xi, yi in zip(X, y)])) / (np.sum([np.power(xi - np.mean(X), 2) for xi in X]))
        self.beta0 = np.mean(y) - (self.beta1 * np.mean(X))

        return
    
    def predict(self, X):
        """
        """

        return self.beta1 * X + self.beta0