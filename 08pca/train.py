import numpy as np
from PCA import PCA  # Custom PCA class from your code
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PCA_custom = PCA(threshold=0.8)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

model_sklearn = LogisticRegression(max_iter=1000)
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Sklearn Logistic Regression Accuracy: {accuracy_sklearn:.6f}")

model_sklearn_customPCA = LogisticRegression(max_iter=1000)
PCA_custom.fit(X_train_copy)

X_train_custom = PCA_custom.transform(X_train_copy)
X_test_custom = PCA_custom.transform(X_test_copy)


model_sklearn_customPCA.fit(X_train_custom, y_train)
y_pred_sklearn_customPCA = model_sklearn_customPCA.predict(X_test_custom)
accuracy_sklearn_customPCA = accuracy_score(y_test, y_pred_sklearn_customPCA)
print(f"Sklearn Logistic Regression with Custom PCA Accuracy: {accuracy_sklearn_customPCA:.6f}")