import sklearn.datasets
import numpy as np
from rasapy.utils.preprocessing import train_test_split
from rasapy.bayesian.nb_classification import NaiveBayesClassification

from sklearn.naive_bayes import MultinomialNB

data = sklearn.datasets.make_classification(n_samples=100, n_features=10, n_informative=4, class_sep=2.0, random_state=115)
X, y = data[0], data[1]
X = np.rint(X).astype(int).clip(0) # Cast continuous classification data into discrete counts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=115)

models = {
    "Multinomial Naive Bayes Classification": NaiveBayesClassification(distribution='multinomial'),
    "Alpha = 0.0 (no smoothing)": NaiveBayesClassification(alpha=0.0),
    "Sklearn MultinomialNB (log smoothing)": MultinomialNB(alpha=0.0)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")