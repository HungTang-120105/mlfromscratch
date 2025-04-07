import numpy as np
from PCA import PCA  # Custom PCA class from your code
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as PCA_sklearn
import matplotlib.pyplot as plt

def plot_pca(X, y, target_names):
    # Only for 2D PCA
    plt.figure(figsize=(8, 6))

    x1 = X[:, 0]
    x2 = X[:, 1]

    # Fix: Update colormap usage
    cmap = plt.colormaps["viridis"]
    
    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=cmap
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of IRIS dataset")
    plt.colorbar(ticks=range(len(target_names)), label="Classes")
    plt.show()

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Create custom PCA object
    pca_custom = PCA(threshold=0.9)
    pca_custom.fit(X)
    X_pca_custom = pca_custom.transform(X)

    # Create PCA object from sklearn
    pca_sklearn = PCA_sklearn(n_components=2)
    pca_sklearn.fit(X)
    X_pca_sklearn = pca_sklearn.transform(X)

    # Plot PCA results
    plot_pca(X_pca_custom, y, target_names)
    plot_pca(X_pca_sklearn, y, target_names)
