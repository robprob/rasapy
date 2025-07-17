from sklearn.datasets import make_classification
from rasapy.utils.preprocessing import train_test_split


from rasapy.trees import RandomForestClassification
from sklearn.ensemble import RandomForestClassifier

data = make_classification(250, 5, n_classes=3, n_clusters_per_class=1, random_state=115)
X, y = data[0], data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "Random Forest": RandomForestClassification(n_estimators=10, random_state=115),
    "Max Features 3": RandomForestClassification(n_estimators=10, max_features=3, random_state=115),
    "Sklearn": RandomForestClassifier(n_estimators=10, random_state=115)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"  Accuracy: {model.score(X_test, y_test):.3f}")