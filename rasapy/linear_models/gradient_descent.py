import numpy as np
from rasapy.linear_models.base_regressor import BaseRegressor

class GradientDescentRegression(BaseRegressor):
    def __init__(self, learning_rate=0.05, epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        """
        Fit weight and bias parameters using gradient descent.
        """
        # Input is of shape m training examples, n features
        m, n = X_train.shape
        
        # Initialize weights and bias parameters
        self.weights = np.zeros(n)
        self.bias = 0.0
        
        # Iterate training epochs
        for epoch in range(self.epochs):
            # Make predictions using existing parameters
            y_pred = self.predict(X_train)
            
            # Calculate residual errors
            errors = y_pred - y_train
            
            # Calculate cost gradient with respect to each feature
            dJ_dw = np.zeros(n)
            # Iterate feature weights
            for w_i in range(n):
                # Parse feature column vector from feature matrix
                X_i = X_train[:, w_i]
                # Accumulate loss gradient, achieving an "average" best gradient direction
                dJ_dw[w_i] = (1 / m) * errors @ X_i
            # Calculate cost gradient with respect to bias parameter
            dJ_db = (1 / m) * np.sum(errors)
            
            # Update model parameters using learning rate and accumulated gradients
            self.weights -= self.learning_rate * dJ_dw
            self.bias -= self.learning_rate * dJ_db