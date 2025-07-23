from rasapy.linear_models import OLSRegression
from rasapy.utils.activation import *

class Perceptron:
    def __init__(self, activation='relu'):
        # Parse activation function
        activation_functions = {
            'linear': linear,
            'relu': relu,
            'leaky_relu': leaky_relu,
            'sigmoid': sigmoid,
            'tanh': tanh
        }
        self.activation = activation_functions.get(activation)
        if self.activation is None:
            self.activation = linear # Fallback to identity function, or linear activation
        
        self.model = OLSRegression() # Base linear model used for predictions
    
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
        