import numpy as np

class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples

    def _gaussian(self, idx, x) -> np.ndarray:
        mean = self.means[idx]
        var = self.var[idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _predict(self, x) -> np.ndarray: 
        posteriors = np.zeros(self.classes.shape[0])
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._gaussian(idx, x)))
            posteriors[idx] = prior + likelihood
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X) -> np.ndarray:
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
        


        

