import numpy as np
from scipy import stats

from rasapy.metrics.classification import accuracy

class KNNClassification:
    """
    Implementation of K-nearest Neighbors Classification, classifying data points by the training data that is numerically closest to it.
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
        Make classification predictions based on the K-nearest data points.
        """
        y_pred = np.zeros(len(X), dtype=int)
        
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
            
            # Make a classification prediction based on the mode of the nearest neighbors
            y_pred[i] = stats.mode(neighbors, keepdims=False).mode # Ties return numerically smaller label
        
        return y_pred
    
    def score(self, X, y_true):
        """
        Make classification predictions and calculate accuracy.
        """
        y_pred = self.predict(X)
        return accuracy(y_true, y_pred)