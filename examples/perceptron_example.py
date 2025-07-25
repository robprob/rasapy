import numpy as np

from sklearn.datasets import make_regression
from rasapy.utils.preprocessing import train_test_split

from rasapy.neural_network import Perceptron

data = make_regression(n_samples=25, n_features=4, bias=15, noise=10.0, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=115)

models = {
    "Linear Perceptron": Perceptron(activation='linear'),
    "Relu": Perceptron(activation='relu'),
    "Leaky Relu": Perceptron(activation='leaky_relu'),
    "Hyperbolic Tangent": Perceptron(activation='tanh')
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"  Output: {np.round(model.predict(X_test), 2)}")