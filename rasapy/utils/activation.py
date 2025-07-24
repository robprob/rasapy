import numpy as np

# Activation Functions
def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(x * alpha, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


# Activation Function Derivatives
def linear_derivative(x):
    x[:] = 1
    return x

def relu_derivative(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

def leaky_relu_derivative(x, alpha=0.01):
    x[x <= 0] *= alpha

def sigmoid_derivative(x):
    x = x * (1 - x)
    return x

def tanh_derivative(x):
    x = 1 - (x**2)