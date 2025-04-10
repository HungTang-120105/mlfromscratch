import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from DBSCAN import DBSCAN  

if __name__ == "__main__":
    X, y_true = make_moons(n_samples=500, noise=0.08, random_state=42)

    db = DBSCAN(eps=0.2, minPts=5)
    db.fit(X)
    y_pred = db.labels  

    ari = adjusted_rand_score(y_true, y_pred)
    print("Adjusted Rand Index (ARI):", ari)

    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow', s=20)
    plt.title(f"My DBSCAN Clustering (ARI = {ari:.2f})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()
