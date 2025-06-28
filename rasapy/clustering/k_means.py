import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001, random_state=None):
        """
        Implementation of KMeans clustering utilizing the Lloyd algorithm.
        """
        self.n_clusters = n_clusters # k number of clusters
        self.max_iter = max_iter # max algorithm iterations
        self.tol = tol # tolerance threshold for convergence
        
        self.clusters = None
        
        # Seed RNG
        if random_state != None:
            np.random.seed(random_state)
    
    def fit(self, X_train):
        """
        Fit K-clusters to the training data, finding a locally optimal solution.
        """
        # Randomly initialize starting clusters
        clusters = self.initialize_clusters(X_train, self.n_clusters)    
        
        previous_inertia = np.inf
        
        # Iterate epochs until completion or convergence
        for epoch in self.max_iter:
            # Assign data points to nearest cluster
            X_clusters = self.assign_clusters(X_train, clusters)
            
            # Update cluster values to the mean values of their assigned data points
            clusters = self.update_clusters(X_train, X_clusters, clusters)
            
            # Calculate overall inertia of current assignment
            current_inertia = self.inertia(X_train, X_clusters, clusters)
            
            # Check for convergence, assessing difference relative to previous inertia
            if abs(current_inertia - previous_inertia) <= self.tol * previous_inertia:
                break
            previous_inertia = current_inertia
        
        # Assign clusters to model
        self.clusters = clusters
        
        
    def initialize_clusters(self, X_train, n_clusters):
        """
        Initialize the starting K-clusters, randomly choosing from distinct training points.
        """
        # Parse distinct training points
        X_train_distinct = np.unique(X_train, axis=0)
        # Choose k data points to be initial clusters
        row_indices = np.random.choice(X_train_distinct.shape[0], size=n_clusters, replace=False)
        return X_train_distinct[row_indices]
    
    def assign_clusters(self, X, clusters):
        """
        Assign each data point to the nearest cluster, returning an array of cluster indices.
        """
        X_clusters = np.empty(len(X), dtype=int)
        for i in range(len(X)):
            point = X[i]
            # Calculate Euclidean distance from each point to existing clusters
            distances = point - clusters # column-wise vectorized subtraction
            distances = np.square(distances) # squared difference
            distances = np.sum(distances, axis=1) # sum squared difference in all cluster dimensions
            distances = np.sqrt(distances) # standardize Euclidean distance with square root
            
            # Assign data point to the closest cluster
            X_clusters[i] = np.argmin(distances)
        
        # Return list of assigned cluster indices
        return X_clusters
    
    def update_clusters(self, X_train, X_clusters, clusters):
        """
        Update cluster values to the mean values of their assigned data points.
        """
        # Iterate clusters
        for i in range(len(clusters)):
            # Parse assigned data points
            points = X_train[np.where(X_clusters == i)]
            # Check for empty cluster, randomly reassigning if necessary
            if points.size == 0:
                clusters[i] = X_train[np.random.choice(X_train.shape[0])]
                continue
            # Assign new cluster as average value in each dimension
            clusters[i] = np.mean(points, axis=0)
        
        return clusters
    
    def inertia(self, X, X_clusters, clusters):
        """
        Calculate total inertia between data points and their assigned clusters.
        """
        # Parse clusters as assigned to each data point
        X_assigned = clusters[X_clusters]
        
        # Calculate squared distance between points and clusters
        squared_distance = np.square(X - X_assigned)
        # Sum total squared distance (inertia)
        total_inertia = np.sum(squared_distance)
        
        return total_inertia