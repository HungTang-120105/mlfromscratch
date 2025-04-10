import numpy as np

from scipy.spatial.distance import cdist

class DBSCAN:
    def __init__(self, eps=0.5, minPts=5):
        self.eps = eps
        self.minPts = minPts
        self.X = None
        self.labels = None
        self.n_clusters = 0
        self.noise = None
        

    def _region_query(self, point_idx):
        distances = cdist([self.X[point_idx]], self.X, 'euclidean')
        neighbors = np.where(distances[0] <= self.eps)[0]
        return neighbors
    
    def _expand_cluster(self, point_idx, neighbors):
        self.labels[point_idx] = self.n_clusters
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not self.visited[neighbor_idx]:
                self.visited[neighbor_idx] = True
                new_neighbors = self._region_query(neighbor_idx)
                
                if len(new_neighbors) >= self.minPts:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = self.n_clusters
            
            i += 1


    def fit(self, X):
        self.X = X
        self.labels = np.full(X.shape[0], -1)
        self.visited = np.zeros(X.shape[0], dtype=bool)
        
        for i in range(X.shape[0]):
            if not self.visited[i]:
                self.visited[i] = True
                neighbors = self._region_query(i)

                if len(neighbors) < self.minPts:
                    self.labels[i] = -1
                    self.noise = np.append(self.noise, i) if self.noise is not None else np.array([i])
                else:
                    self.n_clusters += 1
                    self.labels[i] = self.n_clusters
                    self._expand_cluster(i, neighbors)

    def predict(self, X):
        if self.X is None or self.labels is None:
            raise Exception("Model not fitted yet. Call fit() before predict().")

        distances = cdist(X, self.X, 'euclidean')
        labels = np.full(X.shape[0], -1)

        for i in range(X.shape[0]):
            neighbors = np.where(distances[i] <= self.eps)[0]
            if len(neighbors) >= self.minPts:
                # Đếm số nhãn trong các neighbors, loại trừ nhãn -1 (noise)
                neighbor_labels = self.labels[neighbors]
                core_labels = neighbor_labels[neighbor_labels != -1]
                if len(core_labels) > 0:
                    # Gán nhãn phổ biến nhất trong core_labels
                    labels[i] = np.bincount(core_labels).argmax()
        return labels

                