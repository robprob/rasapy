import numpy as np

from sklearn.datasets import make_classification
from rasapy.utils.preprocessing import train_test_split
from rasapy.metrics import *

from rasapy.neural_network import Perceptron

data = make_classification(n_samples=25, n_features=10, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, random_state=115)

models = {
    "Binary Classification Perceptron": Perceptron(activation='sigmoid', loss='bce'),
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"  Output: {np.round(model.predict(X_test), 2)}")
    print(f"  Accuracy: {np.round(accuracy(y_test, (model.predict(X_test) >= 0.5).astype(int)))}")