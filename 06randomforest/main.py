import numpy as np
from RandomForest import RandomForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from DecisionTree import DecisionTree

# Load the breast cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
X_test_copy = X_test.copy()

# Create and train the RandomForestClassifier from sklearn
sklearn_rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
sklearn_rf.fit(X_train, y_train)
sklearn_accuracy = sklearn_rf.score(X_test, y_test)
print(f"Sklearn Random Forest Accuracy: {sklearn_accuracy:.4f}")



# Create and train the custom RandomForest
rf = RandomForest(n_trees=20, max_depth=10, criterion='ig', min_samples_split=2)
rf.fit(X_train, y_train, list(data.feature_names))
y_pred = rf.predict(X_test)
custom_accuracy = np.mean(y_pred == y_test)
print(f"Custom Random Forest Accuracy: {custom_accuracy:.10f}")



# dt = DecisionTree(max_depth=10, criterion='ig', min_samples_split=2)
# dt.train(X_train, y_train, list(data.feature_names))
# y_pred = dt.predict_all(X_test)
# custom_accuracy = np.mean(y_pred == y_test)
# print(f"Custom Decision Tree Accuracy: {custom_accuracy:.10f}")





