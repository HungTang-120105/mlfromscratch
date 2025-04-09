import numpy as np

class LDA:
    def fit(self, X, y):
        """
        Fit the LDA model to the training data.
        Parameters:
        X : array-like, shape (n_samples, d_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
        W : matrix 

        m_k = mean of class k
        m = mean of all classes
        Sw = within-class scatter matrix
        Sb = between-class scatter matrix

        Sw = sigmama_(k=1..C) sigmama(n in C_K) (x_n - m_k)(x_n - m_k)^T
        Sb = sigmama_(k=1..C) N_k*(m_k - m)(m_k - m)^T

        loss function = argmax(W^T * Sb * W / W^T * Sw * W)

        => Sw^-1 * Sb * W = lambda * W
        => Columns of W are eigenvectors of Sw^-1 * Sb wrt largest eigenvalue lambda
        """
        # Calculate the mean of each class
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        n_features = X.shape[1]

        m = np.zeros((n_classes, n_features))
        for i, cls in enumerate(classes):
            m[i] = np.mean(X[y == cls], axis=0)

        # Calculate the overall mean
        m = np.mean(X, axis=0)

        # Calculate the within-class scatter matrix Sw
        Sw = np.zeros((n_features, n_features))
        for i, cls in enumerate(classes):
            X_cls = X[y == cls]
            Sw += np.cov(X_cls.T) * (X_cls.shape[0] - 1)

        # Calculate the between-class scatter matrix Sb
        Sb = np.zeros((n_features, n_features))
        for i, cls in enumerate(classes):
            N_k = counts[i]
            diff = (m[i] - m).reshape(n_features, 1)
            Sb += N_k * (diff @ diff.T)

        epsilon = 1e-6
        Sw += epsilon * np.eye(Sw.shape[0])

        # Solve the generalized eigenvalue problem Sw^-1 * Sb * W = lambda * W
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigvals)[::-1]
        self.W = eigvecs[:, sorted_indices].real


    def transform(self, X):
        """
        Transform the data using the fitted LDA model.
        Parameters:
        X : array-like, shape (n_samples, d_features)
            Data to transform.
        Returns:
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        return X @ self.W
