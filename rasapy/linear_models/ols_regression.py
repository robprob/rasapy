import numpy as np
from rasapy.linear_models.base_regressor import BaseRegressor

class OLSRegression(BaseRegressor):
    def __init__(self, learning_rate=0.05, epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        """
        Fit weight and bias parameters using the normal equation for Ordinary Least Squares.
        β = (XᵀX)⁻¹Xᵀy
        """
        # Input is of shape m training examples, n features
        m, n = X_train.shape
        
        # Initialize weights and bias parameters
        self.weights = np.zeros(n)
        self.bias = 0.0
        
        # Add a bias column of "1s" to the training features
        # This numerically acts as another "feature" that is always weighted the same
        bias_col = np.ones(m).reshape((-1, 1))
        X_b = np.concatenate((X_train, bias_col), axis=1)
        
        # Solve for Ordinary Least Squares
        beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train
        
        self.weights = beta[:-1]
        self.bias = beta[-1]
        