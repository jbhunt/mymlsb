import numpy as np

class SimpleLinearRegression():
    """
    """

    def __init__(self):
        """
        """

        self.betas = None

        return
    
    def fit(self, X, y):
        """
        """

        xbar = np.mean(X)
        ybar = np.mean(y)
        m = (np.sum([(xi - xbar) * (yi - ybar) for xi, yi in zip(X, y)])) / (np.sum([np.power(xi - xbar, 2) for xi in X]))
        b = np.mean(y) - (m * xbar)
        self.betas = np.array([b, m]).reshape(-1, 1)

        return
    
    def predict(self, X):
        """
        """

        return X @ self.betas[1:] + self.betas[0]
    
class MultipleLinearRegression():
    """
    """

    def __init__(self):
        """
        """

        self.betas = None

        return
    
    def fit(self, X, y):
        """
        """

        X = np.hstack([np.full([X.shape[0], 1], 1), X])
        self.betas = np.linalg.inv(X.T @ X) @ (X.T @ y)

        return
    
    def predict(self, X):
        """
        """

        y = X @ self.betas[1:] + self.betas[0]

        return y
    
class RidgeRegression():
    """
    """

    def __init__(self, lambda_=0.0):
        """
        """

        self.betas = None
        self.lambda_ = lambda_

        return
    
    def fit(self, X, y):
        """
        """

        R = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
        self.betas = np.linalg.inv(R.T @ R + self.lambda_ @ np.eye(R.shape[1])) @ R.T @ y
        self.betas = self.betas.reshape(-1, 1)

        return
    
    def predict(self, X):
        """
        """

        return X @ self.betas[1:] + self.betas[0]