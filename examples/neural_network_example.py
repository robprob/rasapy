from sklearn.datasets import make_regression
from rasapy.utils.preprocessing import train_test_split

from rasapy.neural_network import Dense, NeuralNetwork
from sklearn.neural_network import MLPRegressor

data = make_regression(n_samples=25, n_features=4, n_informative=2, bias=15, noise=10.0, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "Neural Network": NeuralNetwork(
        layers=[
            Dense(n_input=X_train.shape[1], n_output=100, activation='relu'),
            Dense(n_input=100, n_output=1, activation='linear')
            ],
        ),
    "Sklearn MLPRegressor": MLPRegressor(
        hidden_layer_sizes=(100, ),
        activation='relu',
        learning_rate_init=0.01,
        max_iter=1000)
}

for name, model in models.items():
    print(f"{name}:")
    if "Sklearn" in name:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, learning_rate=0.01, epochs=1000)
    print(f"R^2: {model.score(X_test, y_test)}")