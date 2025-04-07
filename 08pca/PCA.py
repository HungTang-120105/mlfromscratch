import numpy as np


class PCA:
    def __init__(self, n_components = None, threshold = 0.95):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.threshold = threshold
    
    def fit(self, X): # X is a 2D array of shape (n_samples, n_features)
        self.mean = np.mean(X, axis = 0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar = False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        if self.n_components is None:
            for i in range(len(eigenvalues)):
                if np.sum(eigenvalues[:i]) / np.sum(eigenvalues) >= self.threshold:
                    self.n_components = i+1
                    break 

        eigenvectors = eigenvectors[:, :self.n_components]
        eigenvalues = eigenvalues[:self.n_components]
        self.components = eigenvectors
        
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
