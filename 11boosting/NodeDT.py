import numpy as np

class NodeDT:
    def __init__(self, X, y, feature_name):
        self.X = X
        self.y = y
        self.feature_name = feature_name
        self.is_leaf = False
        self.label = None
        self.used = []

    def entropy(self):
        """
        Calculate the entropy of the current node.
        """
        if len(self.y) == 0:
            return 0
        _, counts = np.unique(self.y, return_counts=True)
        probs = counts / len(self.y)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def classification_error(self):
        """
        Calculate the classification error of the current node.
        """
        if len(self.y) == 0:
            return 0
        _, counts = np.unique(self.y, return_counts=True)
        probs = counts / len(self.y)
        return 1 - np.max(probs)
    
    def mae(self):
        """
        Calculate the mean absolute error of the current node.
        """
        if len(self.y) == 0:
            return 0
        return np.mean(np.abs(self.y - np.mean(self.y)))
    
    def mse(self):
        """
        Calculate the mean squared error of the current node.
        """
        if len(self.y) == 0:
            return 0
        return np.mean((self.y - np.mean(self.y)) ** 2)
    
