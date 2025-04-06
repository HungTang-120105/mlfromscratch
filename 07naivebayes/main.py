from NaiveBayes import NaiveBayes
import numpy as np
#import NaiveBayes from sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


if __name__ == "__main__":
    # load the breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = 42)

    X_test_copy = X_test.copy()

    # Create and train the GaussianNB from sklearn
    sklearn_nb = GaussianNB()
    sklearn_nb.fit(X_train, y_train)
    sklearn_accuracy = sklearn_nb.score(X_test, y_test)
    print(f"Sklearn GaussianNB Accuracy: {sklearn_accuracy:.6f}")

    # Create and train the custom NaiveBayes
    custom_nb = NaiveBayes()   
    custom_nb.fit(X_train, y_train)
    y_pred = custom_nb.predict(X_test)
    custom_accuracy = np.mean(y_pred == y_test)
    print(f"Custom NaiveBayes Accuracy: {custom_accuracy:.6f}")