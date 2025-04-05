import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from DecisionTree import DecisionTree

# Load the breast cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Create and train the DecisionTreeClassifier from sklearn
sklearn_tree = DecisionTreeClassifier(random_state=42, max_depth = 10)
sklearn_tree.fit(X_train, y_train)
sklearn_accuracy = sklearn_tree.score(X_test, y_test)
print(f"Sklearn Decision Tree Accuracy: {sklearn_accuracy:.4f}")

# Create and train the custom DecisionTree
custom_tree = DecisionTree(max_depth=10, criterion='gini')
custom_tree.train(X_train, y_train, data.feature_names)
y_pred = []
for x_test in X_test:
    y_pred.append(custom_tree.predict(x_test))
custom_accuracy = np.mean(y_pred == y_test)
print(f"Custom Decision Tree Accuracy: {custom_accuracy:.4f}")
