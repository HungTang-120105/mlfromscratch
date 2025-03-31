import numpy as np
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
import os

relative_path = os.path.join(os.path.dirname(__file__), "prostate.txt")

def standardize_regression(X, y):
    """
    Standardize the features and target variable.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)

    X_standardized = (X - X_mean) / X_std
    y_standardized = (y - y_mean) / y_std

    return (X_standardized, X_mean, X_std), (y_standardized, y_mean, y_std)

def main():
    # Load the dataset
    data = np.loadtxt(relative_path, skiprows=1)

    X = data[:, :-1]
    y = data[:, -1]
    y = y.reshape(-1, 1)

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features and target variable
    (X_train,_,_), (y_train, _, _) = standardize_regression(X_train, y_train)
    (X_test, _, _), (y_test, _, _) = standardize_regression(X_test, y_test)

    # Initialize the model
    model = LinearRegression(alpha=0.01, epochs=1000, lambda_=0.1, do_visualize=True)

    model.train(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)


    print("Test score: %f" % model.r2_score(y_pred, y_test))

if __name__ == "__main__":
    main()