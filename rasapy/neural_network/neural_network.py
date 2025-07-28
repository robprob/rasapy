import numpy as np

from rasapy.utils import get_loss, get_score

class NeuralNetwork:
    """
    Customizable feedforward neural network composed of one or more dense layers.
    """
    def __init__(self, layers, loss='mse', random_state=None):
        self.layers = layers # List of layers in the network
        
        # Parse chosen loss and score functions
        self.loss, self.loss_derivative = get_loss(loss)
        self.score_function = get_score(loss)
        
        # Seed RNG
        if random_state != None:
            np.random.seed(random_state)
        
    def forward_prop(self, X):
        """
        Perform a single round of feedforward propagation.
        """
        for layer in self.layers:
            X = layer.forward_prop(X)
        return X
        
    def back_prop(self, dL, learning_rate):
        """
        Perform a single round of back propagation, updating model parameters.
        """
        for layer in reversed(self.layers):
            dL = layer.back_prop(dL, learning_rate)
    
    def fit(self, X_train, y_train, learning_rate=0.01, epochs=1000):
        """
        Fit neural network to training data, propagating for specificied number of epochs.
        """
        # Reshape y_train to match dimensions of y_pred
        y_train = y_train.reshape(-1, 1)
        for epoch in range(epochs):
            # Propagate forward to make predictions
            y_pred = self.forward_prop(X_train)
            # Calculate gradient with respect to chosen loss function
            dL = self.loss_derivative(y_train, y_pred)
            # Propagate backwards to update model parameters
            self.back_prop(dL, learning_rate)
            
    def predict(self, X):
        """
        Make predictions using forward propagation (forward_prop wrapped for consistency).
        """
        return self.forward_prop(X)
    
    def score(self, X, y_true):
        """
        Make predictions and score model based on chosen loss function.
        """
        y_pred = self.predict(X).flatten()
        return self.score_function(y_true, y_pred)