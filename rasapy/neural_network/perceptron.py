import numpy as np

from rasapy.utils.activation import *
from rasapy.metrics import *

from rasapy.utils.parse_input import get_activation, get_loss

class Perceptron:
    """
    Implementation of a simple single-layer perceptron node.
    """
    def __init__(self, activation='relu', loss='mse'):
        self.activation, self.activation_derivative = get_activation(activation)
        self.loss, self.loss_derivative = get_loss(loss)
        
        self.weights = None
        self.bias = None
    
    def param_init(self, n):
        """
        Initialize parameters for a linear model.
        """
        self.weights = np.random.randn(n) * 1e-2
        self.bias = 0.0
    
    def fit(self, X_train, y_train, learning_rate=0.01, epochs=1000):
        """
        Train perceptron using gradient descent, fitting linear parameters to the training data.
        """
        m, n = X_train.shape
        self.param_init(n)
        
        for epoch in range(epochs):
            # Forward pass
            linear_output = X_train @ self.weights + self.bias
            y_pred = self.activation(linear_output)
            
            # Accumulate gradients
            if self.activation == sigmoid and self.loss == binary_cross_entropy:
                grad = y_pred - y_train # Numerically stable shortcut
            else:
                dL = self.loss_derivative(y_train, y_pred) # Loss gradient
                da = self.activation_derivative(linear_output) # Activation gradient
                grad = dL * da # Chain rule
            
            # Calculate gradient with respect to linear parameters
            dL_dw = (X_train.T @ grad) / m
            dL_db = grad.mean()
            
            # Update parameters according to opposite gradient
            self.weights -= learning_rate * dL_dw
            self.bias -= learning_rate * dL_db
    
    def predict(self, X):
        """
        Make predictions based on the model weights and chosen activation function.
        """
        return self.activation(X @ self.weights + self.bias)