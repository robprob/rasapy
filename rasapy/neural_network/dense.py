import numpy as np

from rasapy.utils import *

class Dense:
    def __init__(self, n_input=100, n_output=100, activaton='relu'):
        # Initialize linear parameters for all models in this layer
        self.Weights = np.random.randn(n_input, n_output) * 1e-2
        self.bias = np.zeros((1, n_output))