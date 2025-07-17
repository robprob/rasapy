from sklearn.datasets import make_regression
from rasapy.utils.preprocessing import train_test_split

from rasapy.trees import RandomForestRegression
from sklearn.ensemble import RandomForestRegressor

data = make_regression(250, 5, bias=15, noise=10.0, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "Random Forest": RandomForestRegression(n_estimators=10, random_state=115),
    "Max Features 3": RandomForestRegression(n_estimators=10, max_features=3, random_state=115),
    "Sklearn": RandomForestRegressor(n_estimators=10, random_state=115)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"  R^2: {model.score(X_test, y_test):.3f}")