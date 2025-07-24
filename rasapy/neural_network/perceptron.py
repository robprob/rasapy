import numpy as np

from rasapy.utils.activation import *

class Perceptron:
    def __init__(self, activation='relu'):
        # Parse activation function
        activation_functions = {
            'linear': [linear, linear_derivative],
            'relu': [relu, relu_derivative],
            'leaky_relu': [leaky_relu, leaky_relu_derivative],
            'sigmoid': [sigmoid, sigmoid_derivative],
            'tanh': [tanh, tanh_derivative]
        }
        # Parse chosen activation, otherwise fallback to identity function (aka linear activation)
        self.activation, self.activation_derivative = activation_functions.get(activation, [linear, linear_derivative])
        
        self.weights = None
        self.bias = None
    
    def param_init(self, n):
        """
        Initialize parameters for a linear model.
        """
        self.weights = np.zeros(n)
        self.bias = 0.0
    
    def fit(self, X_train, y_train):
        """
        Fit the associated linear model to training data.
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions based on the model weights and chosen activation function.
        """
        y_pred = self.model.predict(X)
        return self.activation(y_pred)