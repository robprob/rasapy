import numpy as np

def entropy(y_values):
    """
    Calculate entropy as an impurity measure for multi-class classification.
        -∑ pᵢlog2(pᵢ)
        pᵢ = proportion of class i
    """
    # Parse classes and counts from node values
    classes, counts = np.unique(y_values, return_counts=True)
    # Calculate probabilities for each class
    probs = counts / counts.sum()
    # Sum entropy for each class
    # Log of 0 is undefined, set 0log(0) = 0log(1) = 0 instead
    ent = -np.sum(probs * np.log2(np.array([p if p != 0 else 1 for p in probs])))
    return entropy

def gini(y_values):
    """
    Calculate Gini impurity for multi-class classification.
        1 - ∑ pᵢ²
        pᵢ = proportion of class i
    """
    # Parse classes and counts from node values
    classes, counts = np.unique(y_values, return_counts=True)
    # Calculate probabilities for each class
    probs = counts / counts.sum()
    # Sum gini impurity for each class
    gini = 1.0 - np.sum(probs ** 2)
    return gini
    

def binary_entropy(y_values):
    """
    Calculate entropy as an impurity measure for binary classification.
        -pᵢlog2(pᵢ) - (1 - pᵢ)log2(1 - pᵢ)
    """
    p1 = np.mean(y_values)
    p0 = 1 - p1
    # Log of 0 is undefined, set 0log(0) = 0log(1) = 0 instead
    ent = -p1 * np.log2(p1 if p1 != 0 else 1) - p0 * np.log2(p0 if p0 != 0 else 1)
    return ent

def binary_gini(y_values):
    """
    Calculate Gini impurity for binary classification.
        1 - (pᵢ² + (1 - pᵢ)²)
    """
    p1 = np.mean(y_values)
    p0 = 1 - p1
    gini = 1.0 - (p1 ** 2 + p0 ** 2)
    return gini

def accuracy(y_true, y_pred):
    """
    Calculate percentage of correctly classified predictions.
        (1/n) ∑(yᵢ = ŷᵢ)
    """
    acc = np.mean(y_true == y_pred)
    return acc