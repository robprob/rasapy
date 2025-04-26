import numpy as np

def r_squared(y_true, y_pred):
    """
    R^2 = 1 - (RSS / TSS)
    RSS = Residual Sum of Squares
    TSS = Total Sum of Squares
    """
    rss = ((y_true - y_pred)**2).sum()
    tss = ((y_true - y_true.mean())**2).sum()
    
    r2 = 1 - (rss / tss)
    return r2

def mean_squared_error(y_true, y_pred):
    """
    (1/m) * Σ(ŷᵢ - yᵢ)²
    """
    mse = ((y_pred - y_true)**2).mean()
    return mse
    