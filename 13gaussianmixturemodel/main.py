import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture as GMM
from GMD import GMD  

from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true, y_pred):
    """
    Đánh giá accuracy cho bài toán phân cụm bằng ánh xạ Hungarian giữa nhãn thực và nhãn dự đoán
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size

X, y = make_blobs(n_samples=500, centers=3, cluster_std=0.8, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)


model = GMD(n_components=3, max_iter=100, tol=1e-4)
model.fit(X_train)
y_pred = model.predict(X_test)


gmm = GMM(n_components=3, max_iter=100, tol=1e-4)
gmm.fit(X_train)
y_pred_gmm = gmm.predict(X_test)

# custom GMD model evaluation
accuracy = clustering_accuracy(y_test, y_pred)
ari = adjusted_rand_score(y_test, y_pred)
nmi = normalized_mutual_info_score(y_test, y_pred)

print("GMD model:")
print("Accuracy:", accuracy)
print("ARI:", ari)
print("NMI:", nmi)

# sklearn GMM model evaluation
accuracy_gmm = clustering_accuracy(y_test, y_pred_gmm)
ari_gmm = adjusted_rand_score(y_test, y_pred_gmm)
nmi_gmm = normalized_mutual_info_score(y_test, y_pred_gmm)

print("\nSklearn GMM model:")
print("Accuracy:", accuracy_gmm)
print("ARI:", ari_gmm)
print("NMI:", nmi_gmm)
