import numpy as np

class PCA:
    """
    Implementation of linear dimensionality reduction using Singular Value Decomposition.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components # number of components to retain
        
        self.explained_variance_ratio = None # ratios of variance explained by each component
        self.singular_values = None # sqrt(sum of squared distances) for each component
    
    def fit(self, X):
        """
        Project data to a lower dimension space, keeping the desired number of principal components.
        """
        
        # Center data around the origin (subtract mean of each feature)
        mu = np.mean(X, axis=0)
        X -= mu
        
        # Maximize sum of squared distance from origin