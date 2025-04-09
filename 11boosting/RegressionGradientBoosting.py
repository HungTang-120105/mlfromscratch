import numpy as np
from sklearn.tree import DecisionTreeRegressor as DecisionTree

class GradientBoosting:
    def __init__(self, n_estimators=50, learning_rate=0.05, maxdepth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.maxdepth = maxdepth

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.init_pred = np.mean(y)
        y_pred = self.init_pred * np.ones(n_samples)

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            model = DecisionTree(max_depth=self.maxdepth, min_samples_split=2)
            model.fit(X, residuals)
            y_pred += self.learning_rate * model.predict(X)
            self.models.append(model)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.full(n_samples, self.init_pred)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
