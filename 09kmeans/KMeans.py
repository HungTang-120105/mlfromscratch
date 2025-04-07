import numpy as np

class KMeans:
    """"
    Simple implementation of KMeans clustering algorithm.
    Step 1: Choose k random points as initial centroids.
    Step 2: Assign each point to the nearest centroid.
    Step 3: Update the centroids by taking the mean of the points assigned to each centroid.
    Step 4: Repeat steps 2 and 3 until convergence.
    """

    def __init__(self, n_clusters = 3, max_iter = 100, epsilon = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.centroids = None
        self.labels_ = None
        self.vis_elems = {
            "centroids": [],
            "iteration": [],
        }

    def _initialize_centroids(self, X):
        """Randomly initialize centroids from the dataset."""
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]
    
    def _is_converged(self, old_centroids, new_centroids):
        """Check if the centroids have converged."""
        return np.linalg.norm(new_centroids - old_centroids) < self.epsilon
    
    def _assign_labels(self, X):
        """Assign labels to each point based on the nearest centroid."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X):
        """Update centroids by taking the mean of the points assigned to each centroid."""
        new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids
    
    def fit(self, X):
        """Fit the KMeans model to the data."""
        assert type(X) == np.ndarray, "X must be a numpy array"
        assert len(X.shape) == 2, "X must be a 2D array"
        
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            self.labels_ = self._assign_labels(X)
            self.centroids = self._update_centroids(X)

            self.vis_elems["centroids"].append(self.centroids.copy())
            self.vis_elems["iteration"].append(i)

            if self._is_converged(old_centroids, self.centroids):
                break
        print(f"Converged after {i + 1} iterations")

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to."""
        assert self.centroids is not None, "Model has not been fitted yet"
        assert type(X) == np.ndarray, "X must be a numpy array"
        assert len(X.shape) == 2, "X must be a 2D array"
        
        return self._assign_labels(X)
    
    