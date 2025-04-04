import numpy as np
from NodeDT import NodeDT


class DecisionTree:

    _metrics = {
        'ce': '_classification_error',
        'ig': '_information_gain',
    } 
    
    def __init__(self, max_depth = None, criterion = 'ig'):
        self.max_depth = max_depth
        self.criterion = criterion
        if self.criterion not in DecisionTree._metrics:
            self.criterion = 'ig'
        self.num_class = 0
        self.tree = None
        self.threshold = {}

    def _isnumerical(self, feature):
        """
        Check if the feature is numerical.
        """
        return len(np.unique(feature)) >= 100

    def _find_threshold(self, feature, y_train, num_class):
        """
        Find the best threshold for a numerical feature.
        param:
            feature: The feature to find the threshold for.
            y_train: The labels of the training data.
            num_class: The number of classes in the labels.
        return:
            The best threshold for the feature.
        """ 

        assert len(num_class) == 2, "Only binary classification is supported."
        best_threshold = 0.0
        max_exact_classification = 0.0
        is_positive_negative = False
        sorted_feature = sorted(np.unique(feature))
        for i in range(len(sorted_feature) - 1):
            threshold = (sorted_feature[i] + sorted_feature[i + 1]) / 2
            
            left_partition = y_train[feature <= threshold]
            right_partition = y_train[feature > threshold]

            negative_positive = ((len(left_partition[left_partition == 0]) + len(right_partition[right_partition == 1])) 
                                /len(feature))
            positive_negative = ((len(left_partition[left_partition == 1]) + len(right_partition[right_partition == 0]))
                                /len(feature))
            
            is_positive_negative = positive_negative > negative_positive
            choose = positive_negative if is_positive_negative else negative_positive
            if max_exact_classification < choose:
                max_exact_classification = choose
                best_threshold = threshold
        return best_threshold, is_positive_negative

    def _entropy(self, feature, node: NodeDT):
        """
        Compute entropy each partition of specific feature in a given node
        param:
            feature: The feature to compute entropy for.
            node: The node to compute entropy for.
        return:
            an entropy scalar that measure the uncertainty of the feature in the node.
        """
        entropy = 0
        categories = np.unique(feature)
        num_points = len(feature)
        for category in categories:
            num_category = len(feature[feature == category])
            for c in self.num_class:
                num_category_class = len(feature[(feature == category) & (node.y == c)])
                prob = num_category_class / num_category
                if prob > 0:
                    entropy += num_category/num_points * (-prob * np.log2(prob ))
        return entropy
    
    def _information_gain(self, feature, node: NodeDT):
        """
        Compute information gain of a feature in a given node
        param:
            feature: The feature to compute information gain for.
            node: The node to compute information gain for.
        return:
            an information gain scalar that measure the uncertainty of the feature in the node.
        """
        entropy = self._entropy(feature, node)
        return node.entropy() - entropy
    
    def _classification_error(self, feature, node: NodeDT):
        """
        Compute classification error of a feature in a given node
        param:
            feature: The feature to compute classification error for.
            node: The node to compute classification error for.
        return:
            an classification error scalar that measure the uncertainty of the feature in the node.
        """
        return node.classification_error() - self._entropy(feature, node)
    
    def _stop(self, node: NodeDT):
        """
        Check if the node should stop splitting.
        param:
            node: The node to check.
        return:
            True if the node should stop splitting, False otherwise.
        """
        return (len(node.used) == node.X.shape[1]) or (len(node.used) == self.max_depth) or (node.entropy() == 0)

    def _build_dt(self, X_train, root: NodeDT, column_name):
        """
        Algorithm:
            - Start form the root. Find the best feature to split the data.
            - From that best feature, loop through all categories to build subtrees
            ...
            - If entropy/classification error of the node is 0, then stop and move to other subtrees
        param:
            root: The root node of the tree.
            column_name: The name of the feature to split on.
        return:
        """

        N,D = root.X.shape

        best_coef = 0.0
        best_feature = 0

        for d in range(D):
            if column_name[d] in root.used:
                continue
            feature = root.X[:, d]
            func = getattr(self, self._metrics[self.criterion])
            coef = func(feature, root)
            if best_coef < coef:
                best_coef = coef
                best_feature = d
        # after choose the best feature to split,
        # loop through all categories to build subtrees

        feature = root.X[:, best_feature]
        categories = np.unique(X_train[:, best_feature])
        # Để ý categories != np.unique(feature) vì trong quá trình tạo cây, thì feature có thể bị mất đi 

        for category in categories:
            # create a new node for the subtrees
            node = NodeDT(root.X[feature == category], root.y[feature == category], column_name[best_feature])
            node.used = root.used + [column_name[best_feature]]
            setattr(root, 'feature_' + str(category), node)
            setattr(root, 'feature_split', best_feature)
            if not self._stop(node):
                # if the node is not a leaf node, continue to build the tree
                self._build_dt(X_train, node, column_name)
            else:
                node.is_leaf = True
                node.label = 1 if len(node.y[node.y == 1]) > len(node.y[node.y == 0]) else 0

    def _train(self, X_train, y_train, column_name):
        self.tree = NodeDT(X_train, y_train, 'root')
        self._build_dt(X_train, self.tree, column_name)

    def train(self, X_train, y_train, column_name):
        self.num_class = np.unique(y_train)
        _, D = X_train.shape
        for d in range(D):
            feature = X_train[:, d]
            if self._isnumerical(feature):
                threshold, is_positive_negative = self._find_threshold(feature, y_train, self.num_class)
                feature[feature <= threshold] = int(is_positive_negative)
                feature[feature > threshold] = int(not is_positive_negative)
                X_train[:, d] = feature
                self.threshold[d] = (threshold, is_positive_negative)
        self._train(X_train, y_train, column_name)          

    def _predict(self, X_new, node: NodeDT):
        """
        Predict the label of a given sample using the decision tree.
        param:
            X_test: The sample to predict.
            node: The node to predict from.
        return:
            The predicted label of the sample.
        """
        # print(X_new)
        # print(f"Node attributes: {dir(node)}")  # In danh sách thuộc tính của node
        
        
        
        if  (not node.is_leaf):
            # print(f"Current node feature split: {node.feature_split}")
            # print(f"X_new[node.feature_split]: {X_new[node.feature_split]}")
            # print(f"Looking for attribute: feature_{X_new[node.feature_split]}")
            node = getattr(node, 'feature_' + str(X_new[node.feature_split]))
            
            return self._predict(X_new, node)
        return node.label
    
    def predict(self, X_new):
        # First convert numerical feature to categorical feature.
        for key, (threshold, is_positive_negative) in self.threshold.items():
            X_new[key] = int(is_positive_negative) if X_new[key] < threshold else int(not is_positive_negative)
        tree = self.tree
        label = self._predict(X_new, tree)
        return label

    
        

