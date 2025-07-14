import sklearn.datasets
import numpy as np
from rasapy.utils.preprocessing import train_test_split
from rasapy.bayesian.nb_classification import NaiveBayesClassification

from sklearn.naive_bayes import BernoulliNB

data = sklearn.datasets.make_classification(n_samples=100, n_features=10, n_informative=4, class_sep=2.0, random_state=115)
X, y = data[0], data[1]
X = np.array(X >= np.mean(X, axis=0), dtype=int) # Cast continuous classification data into binary features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=115)

models = {
    "Multinomial Naive Bayes Classification": NaiveBayesClassification(distribution='bernoulli'),
    "Alpha = 0.0 (no smoothing)": NaiveBayesClassification(alpha=0.0),
    "Sklearn BernoulliNB (log smoothing)": BernoulliNB(alpha=0.0)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")