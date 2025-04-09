import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from RegressionGradientBoosting import GradientBoosting  # your custom GB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load dữ liệu hồi quy
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Gradient Boosting Regressor
sklearn_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.5, max_depth=5)
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)
print(f"Sklearn GradientBoosting MSE: {sklearn_mse:.4f}")

# Custom Gradient Boosting Regressor
model = GradientBoosting(n_estimators=50, learning_rate=0.5, maxdepth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
custom_mse = mean_squared_error(y_test, y_pred)
print(f"Custom GradientBoosting MSE: {custom_mse:.4f}")
