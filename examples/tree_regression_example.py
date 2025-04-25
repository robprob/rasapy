import sklearn.datasets
from rasapy.trees.tree_regression import TreeRegression
from rasapy.utils.preprocessing import train_test_split

from sklearn.tree import DecisionTreeRegressor

data = sklearn.datasets.make_regression(1000, 4, bias=15, noise=0.0, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "Decision Tree Regression": TreeRegression(),
    "Sklearn": DecisionTreeRegressor()
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"R^2: {model.score(X_test, y_test)}")