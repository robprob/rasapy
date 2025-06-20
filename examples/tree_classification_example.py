import sklearn.datasets
from rasapy.trees.tree_classification import TreeClassification
from rasapy.utils.preprocessing import train_test_split

from sklearn.tree import DecisionTreeClassifier

data = sklearn.datasets.make_classification(1000, 4, n_classes=4, n_clusters_per_class=1, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "Decision Tree Classifier": TreeClassification(random_state=115),
    "Max Depth 3": TreeClassification(max_depth=3, random_state=115),
    "Min Samples Split 3": TreeClassification(min_samples_split=3, random_state=115),
    "Max Features Sqrt": TreeClassification(max_features='sqrt', random_state=115),
    "Max Features 0.5": TreeClassification(max_features=0.5, random_state=115),
    "Max Features 2": TreeClassification(max_features=2, random_state=115),
    "Sklearn": DecisionTreeClassifier(random_state=115)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"  Accuracy: {model.score(X_test, y_test)}")