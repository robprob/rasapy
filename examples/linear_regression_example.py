from sklearn.datasets import make_regression
from rasapy.utils.preprocessing import train_test_split

from rasapy.linear_models import GradientDescentRegression, OLSRegression
from sklearn.linear_model import LinearRegression

data = make_regression(25, 10, bias=15, noise=10.0, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "Gradient Descent": GradientDescentRegression(learning_rate=0.1),
    "OLS": OLSRegression(),
    "Sklearn OLS": LinearRegression()
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"R^2: {model.score(X_test, y_test)}")
    if "Sklearn" in name:
        weights = model.coef_
        bias = model.intercept_
    else:
        weights = model.weights
        bias = model.bias
    print(f"  Weights: {weights}\nBias: {bias}\n")