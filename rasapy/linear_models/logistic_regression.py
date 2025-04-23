import numpy as np
from rasapy.linear_models.base_regressor import BaseRegressor

class LogisticRegression(BaseRegressor):
    def __init__(self, learning_rate=0.05, epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        """
        Fit weight and bias parameters using gradient descent.
        Sigmoid Transformation:
            zᵢ = wxᵢ + b
            ŷᵢ = 1 / (1 + e^(-zᵢ))
        Cost (Negative Log Likelihood):
            J(θ) = Σ -yᵢlog(xᵢ) - (1 - yᵢ)log(1 - xᵢ)
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
    
    def predict(self, X):
        """
        Make probability predictions on feature data using model weights and bias parameters.
        """
        # Call base regressor predict method to make a linear prediction
        z = super().predict(X)
        # Convert linear prediction to sigmoid output
        y_proba = 1 / (1 + np.exp(-z))
        
        return y_proba
    
    def predict_class(self, X, threshold=0.5):
        """
        Make binary class predictions (0, 1) on feature data using model weights and bias parameters.
        """
        # Make probability predictions
        y_proba = self.predict(X)
        
        # Convert probabilities to class predictions
        y_pred = (y_proba >= threshold).astype(int)
        
        return y_pred
    
    def score(self, X, y_true):
        """
        Make binary class predictions on feature data and calculate mean accuracy.
        """
        y_pred = self.predict_class(X)
        
        accuracy = np.mean(y_pred == y_true)
        
        return accuracy