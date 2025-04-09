import numpy as np
from sklearn.tree import DecisionTreeClassifier as DecisionTree

class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=0.05, maxdepth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
        self.maxdepth = maxdepth
    
    def fit(self, X, y):
        
        """
        Step 1: Initialize weights for each sample (w_i = 1/N))
        Step 2: For each estimator:
            - Train a weak learner (Decision Tree)
            - Calculate the error of the weak learner
            r_b = np.sum(w * (y != y_pred)) / np.sum(w)
            - Calculate the alpha (weight) for the weak learner
            alpha = 0.5 * np.log((1 - r_b) / (r_b + 1e-10))
            - Update the weights for the samples
            if y_pred[i] != y[i]:
                w[i] = w[i]  # Increase weight for misclassified samples
            else:
                w[i] *= np.exp(alpha)   # Decrease weight for correctly classified samples
            - Normalize the weights 
        Step 3: Save the weak learner and its alpha
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            # Train a weak learner (Decision Tree)
            model = DecisionTree(max_depth=self.maxdepth, min_samples_split=2, criterion='gini')
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            # Calculate the error of the weak learner
            r_b = np.sum(w * (y != y_pred)) / np.sum(w)

            # Calculate the alpha (weight) for the weak learner
            alpha = 0.5 * np.log((1 - r_b) / (r_b + 1e-10))

            # Update the weights for the samples
            w *= np.exp(self.learning_rate * alpha * (y_pred != y))

            w /= np.sum(w)  

            # Save the weak learner and its alpha
            self.models.append(model)
            self.alphas.append(alpha)
    
    def predict(self, X):
        n_samples = X.shape[0]
        class_scores = np.zeros((n_samples, len(self.classes_)))

        for model, alpha in zip(self.models, self.alphas):
            y_pred = model.predict(X)
            for i, cls in enumerate(self.classes_):
                class_scores[:, i] += alpha * (y_pred == cls)

        return self.classes_[np.argmax(class_scores, axis=1)]
