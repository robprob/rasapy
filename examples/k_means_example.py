from sklearn.datasets import make_blobs
from rasapy.utils.preprocessing import train_test_split

from rasapy.clustering import KMeans
from sklearn.cluster import KMeans as sk_KMeans

X, y = make_blobs(250, centers=5, cluster_std=0.75, random_state=115)
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=115)

models = {
    "K-Means": KMeans(n_clusters=5, random_state=115),
    "Max Iter 500": KMeans(n_clusters=5, max_iter=500, random_state=115),
    "Sklearn": sk_KMeans(n_clusters=5, init='random', n_init=1, random_state=115)
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X_train)
    if "Sklearn" in name:
        score = -model.score(X_test) # sklearn reports inertia as negative (so higher score always better)
    else:
        score = model.score(X_test)
    print(f"  Inertia: {score:.3f}")