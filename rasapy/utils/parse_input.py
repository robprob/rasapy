from .activation import *
from rasapy.metrics import *

def get_activation(function_name):
    activation_functions = {
        'linear': [linear, linear_derivative],
        'relu': [relu, relu_derivative],
        'leaky_relu': [leaky_relu, leaky_relu_derivative],
        'sigmoid': [sigmoid, sigmoid_derivative],
        'tanh': [tanh, tanh_derivative]
    }
    if function_name not in activation_functions:
        raise ValueError(f"Unknown activation function: {function_name}")
    elif function_name is None:
        # Fallback to identity function
        return activation_functions.get('linear')
    
    return activation_functions.get(function_name)
    
def get_loss(function_name):
    loss_functions = {
        'mse': [mean_squared_error, mean_squared_error_derivative],
        'bce': [binary_cross_entropy, binary_cross_entropy_derivative]
    }
    if function_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {function_name}")
    
    return loss_functions.get(function_name)

def get_score(loss_name):
    score_functions = {
        'mse': r_squared
    }
    return score_functions.get(loss_name)