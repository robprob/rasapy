import numpy as np

from rasapy.utils import get_activation

class Dense:
    """
    Single layer of dense (fully-connected) neurons in a feedforward neural network.
    """
    def __init__(self, n_input=100, n_output=100, activation='relu'):
        # Initialize linear parameters for all models in this layer
        self.Weights = np.random.randn(n_input, n_output) * 1e-2
        self.bias = np.zeros((1, n_output))
        
        # Parse chosen activation function
        self.activation, self.activation_derivative = get_activation(activation)
        
        self.X = None # Shape: (n_input), Input (feature) matrix from previous layer
        self.Z = None # Shape: (n_output), Linear output using model weight and bias parameters
        self.A = None # Shape: (n_output), 'Activated' linear output
    
    def forward_prop(self, X):
        """
        Propagate forward, calculating output using linear parameters and chosen activation function.
        """
        self.X = X
        self.Z = self.X @ self.Weights + self.bias
        self.A = self.activation(self.Z)
        return self.A
    
    def back_prop(self, dA, learning_rate):
        """
        Propagate backwards, applying calculus chain rule to update gradients.
        """
        # Gradients with respect to linear output
        dZ = dA * self.activation_derivative(self.Z)
        
        # Gradients with respect to linear parameters
        dW = (self.X.T @ dZ) / self.X.shape[0]
        db = np.mean(dZ, axis=0, keepdims=True)
        
        # Gradients with respect to input features (to pass to previous layer)
        dX = dZ @ self.Weights.T
        
        # Update parameters, moving opposite to gradient
        self.Weights -= dW * learning_rate
        self.bias -= db * learning_rate

        return dX