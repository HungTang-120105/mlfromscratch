import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.model_selection import train_test_split
from AdaBoost import AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Create and train the AdaBoostClassifier from sklearn
base_tree = DecisionTreeClassifier(max_depth=5)
sklearn_model = AdaBoostClassifier(estimator=base_tree,
                         n_estimators=50,
                         learning_rate=0.5)
sklearn_model.fit(X_train, y_train)
sklearn_accuracy = sklearn_model.score(X_test, y_test)
print(f"Sklearn AdaBoost Accuracy: {sklearn_accuracy:.4f}")

# Create and train the custom AdaBoost custom
model = AdaBoost(n_estimators=50, learning_rate=0.5, maxdepth=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
custom_accuracy = np.mean(y_pred == y_test)
print(f"Custom AdaBoost Accuracy: {custom_accuracy:.4f}")


