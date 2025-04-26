import numpy as np

def entropy(y_values):
    """
    Calculate entropy as an impurity measure for binary classification.
        -yᵢlog2(xᵢ) - (1 - yᵢ)log2(1 - xᵢ)
    """
    p1 = np.mean(y_values)
    p0 = 1 - p1
    # Log of 0 is undefined, set 0log(0) = 0log(1) = 0 instead
    ent = -p1 * np.log2(p1 if p1 != 0 else 1) - p0 * np.log2(p0 if p0 != 0 else 1)
    return ent

def gini(y_values):
    """
    Calculate Gini impurity for binary classification.
        1 - (xᵢ^2 + (1 - xᵢ)^2)
    """
    p1 = np.mean(y_values)
    p0 = 1 - p1
    return 1.0 - (p1 ** 2 + p0 ** 2)

def accuracy(y_true, y_pred):
    """
    Calculate percentage of correctly classified predictions.
    """
    acc = np.mean(y_true == y_pred)
    return acc