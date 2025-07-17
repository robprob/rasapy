from sklearn.datasets import make_blobs
from rasapy.utils.preprocessing import train_test_split

from rasapy.neighbors import KNNClassification
from sklearn.neighbors import KNeighborsClassifier

X, y = make_blobs(250, centers=5, cluster_std=1.5, random_state=115)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "K-nearest Neighbors Classifier (k=5)": KNNClassification(k=5),
    "Sklearn (k=5)": KNeighborsClassifier(n_neighbors=5, metric='euclidean')
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train, y_train)
    print(f"  Accuracy: {model.score(X_test, y_test):.3f}")