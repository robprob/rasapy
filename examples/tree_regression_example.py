import sklearn.datasets
from rasapy.trees.tree_regression import TreeRegression
from rasapy.utils.preprocessing import train_test_split

from sklearn.tree import DecisionTreeRegressor

data = sklearn.datasets.make_regression(1000, 4, bias=15, noise=0.0, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "Decision Tree Regression": TreeRegression(random_state=115),
    "Max Depth 3": TreeRegression(max_depth=3, random_state=115),
    "Min Samples Split 5": TreeRegression(min_samples_split=5, random_state=115),
    "Max Features Sqrt": TreeRegression(max_features='sqrt', random_state=115),
    "Max Features 0.5": TreeRegression(max_features=0.5, random_state=115),
    "Max Features 2": TreeRegression(max_features=2, random_state=115),
    "Sklearn": DecisionTreeRegressor(random_state=115)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"R^2: {model.score(X_test, y_test)}")