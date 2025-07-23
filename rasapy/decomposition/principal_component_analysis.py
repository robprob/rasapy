import numpy as np

class PCA:
    """
    Implementation of linear dimensionality reduction using Singular Value Decomposition.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components # number of components to retain
        
        self.mu = None # Means of each input feature
        self.components = None # Principal eigenvector components
        
        self.explained_variance_ratio = None # ratios of variance explained by each component
        self.singular_values = None # sqrt(SS(distance from origin)) for each component
    
    def fit(self, X_train):
        """
        Project data to a lower dimension space, keeping the desired number of principal components.
        """
        # Center data around the origin (subtract mean of each feature)
        self.mu = np.mean(X_train, axis=0)
        Xc = X_train - self.mu # X centered
        
        # Perform Signular Value Decomposition of input matrix
        U, S, Vh = np.linalg.svd(Xc, full_matrices=True, compute_uv=True)
        self.singular_values = S
        
        # Parse number of components to retain, otherwise all
        k = self.n_components or Xc.shape[0]
        # Parse chosen number of eigenvector components
        self.components = Vh[:k]
        
        # Calculate proportion of variance explained by each component
        """
        # Singular Values = sqrt(SS(distance from origin))
        # Eigenvalues = SS(distance from origin) / (n - 1)
        """
        SS = (S**2) # Sum of squares
        eigenvalues = SS / (Xc.shape[0] - 1) # Component eigenvalues
        self.explained_variance_ratio = eigenvalues[:k] / eigenvalues.sum()
        
        return self # Allows chaining into transform
    
    def transform(self, X):
        """
        Transform input matrix according to fitted eigenvector components.
        """
        # Center data
        Xc = X - self.mu
        # Matrix multiply by components
        X_transformed = Xc @ self.components
        return X_transformed
    
    def fit_transform(self, X):
        """
        Perform PCA fit and transformation at once.
        """
        return self.fit(X).transform(X)