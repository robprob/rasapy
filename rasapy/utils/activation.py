import numpy as np

# Activation Functions
def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(x * alpha, x)

def sigmoid(x):
    x = np.clip(x, -100, 100) # Prevent overflow
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


# Activation Function Derivatives
def linear_derivative(x):
    return np.ones_like(x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - (x**2)