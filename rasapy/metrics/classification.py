import numpy as np

# Cost and Evaluation Metrics
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
    ent = -np.sum(probs * np.log2(np.where(probs > 0, probs, 1)))
    return ent

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

def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    """
    Compute binary cross entropy (BCE) between predicted probabilities and true labels.
        BCE = -[ylog(pᵢ) + (1 - y) log(1 - pᵢ)]
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clip predictions to prevent log(0)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


# Derivatives
def binary_cross_entropy_derivative(y_true, y_pred, epsilon=1e-10):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Clip predictions to prevent log(0)
    dL_da = (y_pred - y_true) / (y_pred * (1 - y_pred))
    return dL_da