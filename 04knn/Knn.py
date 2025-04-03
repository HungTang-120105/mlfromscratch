import numpy as np
from scipy.spatial.distance import cdist

_metrics = {
    'euclidean': '_l2_distance',
    'manhattan': '_l1_distance',
    'cosine': '_cosine_similarity',
}

class KNN:
    def __init__(self, X, y, k, metric='euclidean'):
        self.k = k
        print(f"X type: {type(X)}, y type: {type(y)}")
        assert type(X) == np.ndarray, "X must be a numpy array"
        assert type(y) == np.ndarray, "y must be a numpy array"
        self.X = X
        self.y = y
        self.metric = metric
        self.classes = np.unique(y)

    def _l1_distance(self, X_new):
        return cdist(X_new, self.X, 'cityblock')
    
    def _l2_distance(self, X_new):
        return cdist(X_new, self.X, 'euclidean')
    
    def _cosine_similarity(self, X_new):
        return cdist(X_new, self.X, 'cosine')
    
    def predict(self, X_new):
        assert type(X_new) == np.ndarray, "X_new must be a numpy array"
        assert X_new.shape[1] == self.X.shape[1], "X_new must have the same number of features as X"
        
        if self.metric not in _metrics:
            self.metric = 'euclidean'

        # Calculate distances
        func = getattr(self, _metrics[self.metric]) # Get the distance function based on the metric
        dist = func(X_new)
        
        # Get the k nearest neighbors
        knn_indices = np.argsort(dist, axis=1)[:, :self.k]
        
        # Get the labels of the k nearest neighbors
        knn_labels = self.y[knn_indices]
        knn_labels = knn_labels.astype(int)
        # Predict the label by majority voting
        predictions = np.array([np.bincount(labels).argmax() for labels in knn_labels])
        
        return predictions
