from abc import ABC, abstractmethod
import numpy as np
from rasapy.metrics.regression import r_squared

class BaseRegressor:
    def __init__(self):
        self.weights = None
        self.bias = 0.0
    
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Fit model weights and bias parameters using model-specific training algorithm.
        """
        pass
    
    def predict(self, X):
        """
        Make predictions on feature data using model weights and bias parameters.
        """
        y_pred = X @ self.weights + self.bias
        return y_pred
    
    def score(self, X, y_true):
        """
        Make predictions on feature data and calculate coefficient of determination (R^2).
        """
        y_pred = self.predict(X)
        r2 = r_squared(y_true, y_pred)
        
        return r2