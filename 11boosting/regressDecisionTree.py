import numpy as np
from DecisionTree import DecisionTree
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error



# Load dữ liệu
data = load_diabetes()
X, y = data.data, data.target

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

custom_decision_tree = DecisionTree(max_depth=6, min_samples_split=2,criterion='mse')
custom_decision_tree.train(X_train, y_train, data.feature_names)
y_pred = custom_decision_tree._predict_all_regression(X_test.copy())
print("Mean Absolute Error custom decision tree: ", mean_absolute_error(y_test, y_pred))
print("Mean squared Error custom decision tree: ", np.mean((y_test - y_pred) ** 2))


# Tạo mô hình DecisionTreeRegressor từ sklearn
sklearn_model = DecisionTreeRegressor(max_depth=6, min_samples_split=2, criterion='squared_error')
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
print("Mean Absolute Error sklearn decision tree: ", mean_absolute_error(y_test, y_pred_sklearn))
print("Mean squared Error sklearn decision tree: ", np.mean((y_test - y_pred_sklearn) ** 2))

