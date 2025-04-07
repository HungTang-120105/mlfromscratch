import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from KMeans import KMeans

# Tạo dữ liệu giả lập gồm 3 cụm (clusters)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Khởi tạo và huấn luyện mô hình KMeans
model = KMeans(n_clusters=3, max_iter=100, epsilon=1e-4)
model.fit(X)
y_pred = model.predict(X)

# Vẽ kết quả
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("KMeans Clustering (Custom Implementation)")
plt.show()
