import numpy as np
from rasapy.metrics.regression import r_squared

class TreeRegression:
    def __init__(self, max_depth=None, max_features=None, random_state=None):
        self.root = None # Root node of tree
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit decision tree nodes and parameters using training data.
        """
        # Instantiate root node
        self.root = TreeNode(indices=range(X_train.shape[0]))

        # Recursively instantiate tree on training data
        self.root.recursive_split(X_train, y_train)
    
    def predict(self, X):
        """
        Make a regression prediction based on which leaf node is reached.
        """
        m, n = X.shape
        y_pred = np.empty(m)
        for i in range(m):
            # Recursively traverse down decision nodes, returning a prediction at leaf node
            y_pred[i] = self.root.traverse(X[i])
            
        return y_pred
    
    def score(self, X, y_true):
        """
        Make predictions on feature data and calculate coefficient of determination (R^2).
        """
        y_pred = self.predict(X)
        r2 = r_squared(y_true, y_pred)
        
        return r2
    


class TreeNode:
    def __init__(self, indices=None, depth=0):
        self.indices = np.array(indices) # List of training indices at node
        self.depth = depth # Current depth of node (root = 0)
        self.feature = None # Feature index used to split node
        self.split_value = None # Value used to split node
        self.left_node = None # Left branch TreeNode
        self.right_node = None # Right branch TreeNode
        self.prediction = None # Prediction output at leaf node

    def best_split(self, X_train, y_train):
        """
        Determine best feature/value split (lowest cost) for current node's indices.
        """
        m, n = X_train.shape
        
        best_feature = None
        best_value = None
        lowest_cost = np.inf
        
        # Iterate features
        for col in range(n):
            # Parse current feature column
            feat_col = X_train[self.indices, col]
            
            # Obtain indices that would sort feature values
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
                
                # Calculate weighted variance of resulting split
                left_w, right_w = len(left_indices) / len(sorted_indices), len(right_indices) / len(sorted_indices)
                weighted_var = (np.var(y_train[left_indices]) * left_w) + (np.var(y_train[right_indices]) * right_w)
        
                # If lower than current lowest cost, update best feature/value split
                if weighted_var < lowest_cost:
                    best_feature = col
                    # Split on average value between ordered feature values
                    best_value = (feat_col[sorted_indices[i - 1]] + feat_col[i]) / 2
                    lowest_cost = weighted_var
            
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

        # Instantiate new branch nodes
        self.left_node = TreeNode(left_indices)
        self.right_node = TreeNode(right_indices)

    def recursive_split(self, X_train, y_train):
        """
        Continually recursively split node until a leaf is reached.
        """
        # Check for leaf node (only 1 index)
        if len(self.indices) == 1:
            # Calculate and set regression prediction as average of values at this node
            self.prediction = np.mean(y_train[self.indices])
            return
        
        # Split this node
        self.split_node(X_train, y_train)
        
        # Split left and right nodes recursively
        if self.left_node and self.right_node:
            self.left_node.recursive_split(X_train, y_train)
            self.right_node.recursive_split(X_train, y_train)

    def traverse(self, x_i):
        """
        Given a sample, recursively traverse down decision nodes, returning a prediction at leaf node.
        """
        # Check for leaf node with a valid prediction
        if self.prediction != None:
            return self.prediction

        # Parse relevant feature value for split
        x_ij = x_i[self.feature]

        # Use feature value to determine which branch to traverse next
        if x_ij < self.split_value:
            return self.left_node.traverse(x_i)
        else:
            return self.right_node.traverse(x_i)
        