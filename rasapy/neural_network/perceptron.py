from rasapy.linear_models import OLSRegression
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
        function = activation_functions.get(activation)
        if function is None:
            self.activation = linear # Fallback to identity function, aka linear activation
            self.derivative = linear_derivative
        else:
            self.activation = function[0]
            self.derivative = function[1]
        
        self.model = OLSRegression() # Base linear model used for predictions
        self.weights = None
        self.bias = None
    
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