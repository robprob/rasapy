import numpy as np
from rasapy.trees.tree_regression import TreeRegression
from rasapy.metrics.regression import r_squared

class RandomForestRegression:
    """
    Implementation of an ensembled random forest of regression trees.
    """
    def __init__(self, n_estimators=100, bootstrap=True, max_samples=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self.forest = [] # list of estimators
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples # int or float, default=None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        # Seed RNG
        np.random.seed(random_state)
        
    def fit(self, X_train, y_train):
        """
        Instantiate and grow the forest, fitting it to the training data.
        """
        m, n = X_train.shape
        
        # Parse max_samples parameter
        max_samples = self.max_samples
        if max_samples is None:
            max_samples = m
        elif isinstance(max_samples, int):
            pass
        elif isinstance(max_samples, float):
            max_samples = max(1, int(max_samples * m))
        else:
            raise ValueError(f"Invalid parameter input for max_samples: {max_samples}")
        
        # Instantiate forest
        forest = np.array([TreeRegression(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          max_features=self.max_features) for _ in range(self.n_estimators)])
        
        # Iterate estimators
        for tree in forest:
            # Bootstrap training samples
            if self.bootstrap:
                subset = np.random.choice(m, max_samples, replace=True)
                X_subset = X_train[subset]
                y_subset = y_train[subset]
            else:
                X_subset = X_train
                y_subset = y_train
            
            # Fit tree to training subset
            tree.fit(X_subset, y_subset)
            
        # Assign forest to the model
        self.forest = forest
        
    def predict(self, X):
        """
        Make a regression prediction based on average predictions of the ensembled trees
        """
        # Iterate estimators, making predictions
        y_pred = np.array([tree.predict(X) for tree in self.forest])
        
        # Return averaged predictions
        return np.mean(y_pred, axis=0)
    
    def score(self, X, y_true):
        """
        Make predictions on feature data and calculate coefficient of determination (R^2).
        """
        y_pred = self.predict(X)
        r2 = r_squared(y_true, y_pred)
        
        return r2