import numpy as np
from scipy import stats
from rasapy.metrics.classification import entropy, gini, accuracy

class TreeClassification:
    """
    Implementation of a decision tree for classification of a single binary output variable.
    """
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self.root = None
        
        # Parse criterion to assign the correct cost function
        if criterion == "entropy":
            criterion = entropy
        elif criterion == "gini":
            criterion = gini
        else:
            raise ValueError("fInvalid criterion: {criterion}")
        
        self.params = {
            "criterion": criterion, # criterion for evaluating quality of a split
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features, # int, float or {“sqrt”, “log2”}, default=None, 
            "random_state": random_state
        }
        # Seed RNG
        np.random.seed(random_state)
    
    def fit(self, X_train, y_train):
        """
        Fit decision tree nodes and parameters using training data.
        """
        # Instantiate root node
        self.root = TreeNode(indices=range(X_train.shape[0]), depth=0, params=self.params)

        # Recursively instantiate tree on training data
        self.root.recursive_split(X_train, y_train)
    
    def predict(self, X):
        """
        Make a classification prediction based on the majority of values in node reached.
        """
        m, n = X.shape
        y_pred = np.empty(m)
        for i in range(m):
            # Recursively traverse down decision nodes, returning a prediction at leaf node
            y_pred[i] = self.root.traverse(X[i])
        
        y_pred = (y_pred >= 0.5).astype(int)
            
        return y_pred
    
    def score(self, X, y_true):
        """
        Make predictions on feature data and calculate accuracy.
        """
        y_pred = self.predict(X)
        acc = accuracy(y_true, y_pred)
        
        return acc


class TreeNode:
    def __init__(self, indices, depth, params):
        self.indices = np.array(indices)   # List of training indices at node
        self.params = params               # Parameter dictionary passed down from TreeRegression
        
        self.depth = depth                 # Current depth of node (root = 0)
        self.feature = None                # Feature index used to split node
        self.split_value = None            # Value used to split node
        self.left_node = None              # Left branch TreeNode
        self.right_node = None             # Right branch TreeNode
        self.prediction = None             # Prediction output at leaf node

    def best_split(self, X_train, y_train):
        """
        Determine best feature/value split (lowest cost) for current node's indices.
        """
        m, n = X_train.shape
        
        best_feature = None
        best_value = None
        lowest_cost = np.inf
        
        # Parse max_features parameter
        max_features = self.params["max_features"]
        if max_features is None:
            max_features = n
        elif isinstance(max_features, int):
            pass
        elif isinstance(max_features, float):
            max_features = max(1, int(max_features * n))
        elif max_features == "sqrt":
            max_features = max(1, int(np.sqrt(n)))
        elif max_features == "log2":
            max_features = max(1, int(np.log2(n)))
        else:
            raise ValueError(f"Invalid parameter input for max_features: {max_features}")
        
        # Create a random permutation of feature indices
        # Even if max_features = None, random feature permutation improves randomness/generalizability
        feature_indices = np.random.permutation(np.arange(n))
        # Parse feature indices
        feature_indices = feature_indices[:max_features]
        
        # Iterate features
        for col in feature_indices:
            # Parse current feature column
            feat_col = X_train[self.indices, col]
            
            # Obtain sample indices that would sort feature values
            sorted_indices = np.argsort(feat_col)
            
            # Iterate sorted data, locating the best split
            for i in sorted_indices[1:]:
                # Determine split value and which indices split left vs right
                split_value = feat_col[i]
                left_indices = np.where(feat_col < split_value)[0]
                right_indices = np.where(feat_col >= split_value)[0]
                
                # Convert left and right indices back into global training data indices
                left_indices = self.indices[left_indices]
                right_indices = self.indices[right_indices]
                
                # Parse chosen cost function
                cost_function = self.params["criterion"]
                # Calculate weighted cost of resulting split
                left_w, right_w = len(left_indices) / len(sorted_indices), len(right_indices) / len(sorted_indices)
                weighted_entropy = (cost_function(y_train[left_indices]) * left_w) + (cost_function(y_train[right_indices]) * right_w)
        
                # If lower than current lowest cost, update best feature/value split
                if weighted_entropy < lowest_cost:
                    best_feature = col
                    # Split on average value between ordered feature values
                    best_value = (feat_col[sorted_indices[i - 1]] + feat_col[i]) / 2
                    lowest_cost = weighted_entropy
        
        return best_feature, best_value
        
    def split_node(self, X_train, y_train):
        """
        Split node using best feature/value split.
        """
        # Determine best feature/value split (lowest cost) for current node's indices
        best_feature, best_value = self.best_split(X_train, y_train)
        # Assign split information to this node
        self.feature = best_feature
        self.split_value = best_value
        
        # Parse entire feature column from training set
        feat_col = X_train[:, best_feature]
        
        # Determine indices for left/right splits, then narrow to indices at current node (intersect)
        left_indices = np.intersect1d(np.where(feat_col < best_value)[0], self.indices)
        right_indices = np.intersect1d(np.where(feat_col >= best_value)[0], self.indices)
        
        # Check for minimum allowed samples per leaf
        if (len(left_indices) < self.params["min_samples_leaf"]) or (len(right_indices) < self.params["min_samples_leaf"]):
            self.set_prediction(y_train[self.indices])
            return
        # Check for maximum allowed depth
        if self.depth == self.params["max_depth"]:
            self.set_prediction(y_train[self.indices])
            return
        # Check for empty split indices
        if len(left_indices) == 0 or len(right_indices) == 0:
            self.set_prediction(y_train[self.indices])
            return

        # Instantiate new branch nodes
        self.left_node = TreeNode(left_indices, self.depth + 1, self.params)
        self.right_node = TreeNode(right_indices, self.depth + 1, self.params)

    def recursive_split(self, X_train, y_train):
        """
        Continually recursively split node until a leaf is reached.
        """
        # Check for completely pure node
        if np.all(y_train[self.indices] == y_train[self.indices][0]): # All values == first index
            self.set_prediction(y_train[self.indices])
            return
        
        # Check minimum necessary samples to split
        if len(self.indices) < self.params["min_samples_split"]:
            self.set_prediction(y_train[self.indices])
            return
        
        # Split this node
        self.split_node(X_train, y_train)
        
        # Split left and right nodes recursively
        if self.left_node and self.right_node:
            self.left_node.recursive_split(X_train, y_train)
            self.right_node.recursive_split(X_train, y_train)
    
    def set_prediction(self, values):
        """
        Calculate and set classification prediction as most common output (mode) of this node.
        """
        self.prediction = stats.mode(values, keepdims=False).mode

    def traverse(self, x_i):
        """
        Given a sample, recursively traverse down decision nodes, returning a prediction at leaf node.
        """
        # Check for leaf node with a valid prediction
        if self.prediction is not None:
            return self.prediction

        # Parse relevant feature value for split
        x_ij = x_i[self.feature]

        # Use feature value to determine which branch to traverse next
        if x_ij < self.split_value:
            return self.left_node.traverse(x_i)
        else:
            return self.right_node.traverse(x_i)