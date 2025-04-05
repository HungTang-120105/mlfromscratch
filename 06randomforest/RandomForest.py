import numpy as np
from DecisionTree import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, criterion='ig', min_samples_split=2):
        self.n_trees = n_trees
        self.trees = []
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split  
    
    def bootstrap(self, X, y):
        """
        Create a bootstrap sample from the dataset.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, int(n_samples*0.8), replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y, column_name):
        """
        Train the Random Forest model.
        """
        self.trees = []
        for _ in range(self.n_trees):
            X_bootstrap, y_bootstrap = self.bootstrap(X, y)
            tree = DecisionTree(max_depth=self.max_depth, criterion=self.criterion, min_samples_split=self.min_samples_split)
            # if self.n_features is None:
            #     self.n_features = X.shape[1]
            # feature_indices = np.random.choice(X.shape[1], self.n_features, replace=False)
            # tree.train(X_bootstrap[:, feature_indices], y_bootstrap, feature_indices)
            # self.trees.append(tree)

            tree.train(X_bootstrap, y_bootstrap, column_name= column_name)
            self.trees.append(tree)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict_all(X.copy()) for tree in self.trees]) 
        # Truyền vào X.copy(), nếu truyền vào X thì cây sẽ k học được
        # vì X sẽ bị thay đổi trong quá trình dự đoán
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
   

 



    