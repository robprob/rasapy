import numpy as np

from rasapy.metrics.regression import r_squared

class KNNRegression:
    """
    Implementation of K-nearest Neighbors Regression, making predictions based on the training data that is numerically closest.
    """
    def __init__(self, k=5):
        self.k = k
        self.X = None # Training data points ARE the model parameters
        self.y = None
    
    def fit(self, X_train, y_train):
        """
        Fit KNN model to the training data.
        """
        # Assign training data to the model
        self.X = X_train
        self.y = y_train
    
    def predict(self, X):
        """
        Make regression predictions based on the K-nearest data points.
        """
        y_pred = np.zeros(len(X), dtype=float)
        
        # Iterate points in X
        for i in range(len(X)):
            point = X[i]
            
            # Calculate Euclidean distances from point to all training data points
            distances = point - self.X # column-wise vectorized difference
            distances = np.square(distances) # squared difference
            distances = np.sum(distances, axis=1) # sum squared difference in all point dimensions
            distances = np.sqrt(distances) # standardize Euclidean distance with square root
            
            # Parse classes of the k-nearest neighbors
            neighbors = self.y[np.argpartition(distances, self.k)[:self.k]]
            
            # Make a regression prediction based on the mean of nearest neighbors
            y_pred[i] = np.mean(neighbors)
        
        return y_pred
    
    def score(self, X, y_true):
        """
        Make regression predictions and calculate coefficient of determination (R^2).
        """
        y_pred = self.predict(X)
        return r_squared(y_true, y_pred)