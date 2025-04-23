import sklearn.datasets
from rasapy.linear_models.logistic_regression import LogisticRegression
from rasapy.utils.preprocessing import train_test_split

from sklearn.linear_model import LogisticRegression as ScikitLogisticRegression

data = sklearn.datasets.make_classification(25, 10, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Gradient Descent": LogisticRegression(),
    "Sklearn LogReg": ScikitLogisticRegression(penalty=None)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test)}")
    if "Sklearn" in name:
        weights = model.coef_
        bias = model.intercept_
    else:
        weights = model.weights
        bias = model.bias
    print(f"Weights: {weights}\nBias: {bias}\n")