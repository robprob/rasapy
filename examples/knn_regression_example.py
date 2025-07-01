from rasapy.neighbors.knn_regression import KNNRegression
from rasapy.utils.preprocessing import train_test_split

from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor

data = make_regression(250, 10, bias=15, noise=10.0, random_state=115)
X, y = X, y = data[0], data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "K-nearest Neighbors Regressor (k=5)": KNNRegression(k=5),
    "k=3": KNNRegression(k=3),
    "k=8": KNNRegression(k=8),
    "Sklearn (k=5)": KNeighborsRegressor(n_neighbors=5, metric='euclidean')
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"  R^2: {model.score(X_test, y_test):.3f}")