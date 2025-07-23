from sklearn.datasets import make_regression

from rasapy.decomposition import PCA
from sklearn.decomposition import PCA as sk_PCA

data = make_regression(n_samples=25, n_features=4, n_informative=2, bias=15, noise=10.0, random_state=115)
X = data[0]

models = {
    "PCA": PCA(),
    "Sklearn PCA": sk_PCA()
}

for name, model in models.items():
    print(f"{name}:")
    model.fit(X)
    if "Sklearn" in name:
        explained_variance_ratio = model.explained_variance_ratio_
        singular_values = model.singular_values_
    else:
        explained_variance_ratio = model.explained_variance_ratio
        singular_values = model.singular_values
    print(f"  Proportion of Explained Variance: {explained_variance_ratio}\n  Singular Values: {singular_values}\n")