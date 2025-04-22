import numpy as np

def train_test_split(X, y, test_size=None, random_state=None, shuffle=True):
    """
    Splits feature and target data into train and test sets for model training and evaluation
    """
    m = X.shape[0]
    # Shuffle input arrays
    if shuffle == True:
        rng = np.random.default_rng(random_state)
        index_vals = rng.permutation(m)
        X, y = X[index_vals], y[index_vals]
    
    # Determine index to split on
    if isinstance(test_size, int):
        # test_size integer represents absolute test set size
        split_index = test_size
    elif isinstance(test_size, float):
        # test_size float represents test set proportion
        split_index = int(m * test_size)
    else:
        split_index = 0
    
    X_train = X[split_index:]
    X_test = X[:split_index]
    
    y_train = y[split_index:]
    y_test = y[:split_index]
    
    return X_train, X_test, y_train, y_test