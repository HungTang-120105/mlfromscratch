import os
import sys
# Thêm đường dẫn thư mục cha (project/) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from nn_components.activations import _softmax
from nn_components.losses import CrossEntropy


class SoftmaxRegression:
    def __init__(self, feature_dim: int, num_classes:int, optimizer: object, loss_func:object = CrossEntropy):
        """
        Softmax Regression model.
        Parameters:
        feature_dim: int, number of features
        num_classes: int, number of classes
        optimizer: object, optimizer for training (e.g., SGD, Adam)
        loss_func: object, loss function (default is CrossEntropy)
        """
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.w = np.random.normal(size = (feature_dim, num_classes))  # weights

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        x: ndarray, input data of shape (num_samples, num_features)
        z = x.dot(w)
        A = softmax(z), shape = (num_samples, num_classes)
        """

        Z = X.dot(self.w)
        A = _softmax(Z)

        return A
    
    def backward(self, y:np.ndarray, y_hat:np.ndarray, X:np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the weights.
        dL/dW = X.T.dot(y_hat - y) / m
        where m is the number of samples.
        x : ndarray, input data of shape (num_samples, num_features)
        y : ndarray, true labels (one-hot encoded), shape = (num_samples, num_classes)
        y_hat : ndarray, predicted probabilities, shape = (num_samples, num_classes)
        gradient: shape = (num_features, num_classes)
        """

        dz = self.loss_func.backward(y, y_hat)  # gradient of the loss w.r.t. the output of softmax
        dw = X.T.dot(dz) 
        dw = self.optimizer.minimize(dw)  # apply the optimizer to the gradient
        self.w -= dw  # update the weights
        
    def __call__(self, X: np.ndarray):
        return self._forward(X)
    
    def predict(self, X_test: np.ndarray):
        return np.argmax(self._forward(X_test), axis = 1)
    
