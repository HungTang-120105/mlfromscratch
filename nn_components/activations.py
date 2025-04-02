import numpy as np

def _sigmoid(z):
    """"
    sigmoid activation function.
    g(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def _sigmoid_grad(z):
    """
    Gradient of the sigmoid function.
    g'(z) = g(z) * (1 - g(z))
    """
    return _sigmoid(z) * (1 - _sigmoid(z))

def _tanh(z):
    """
    tanh activation function.
    g(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    """
    return np.tanh(z)

def _tanh_grad(z):
    """
    Gradient of the tanh function.
    g'(z) = 1 - g(z)^2
    """
    return 1 - np.tanh(z)**2

def _relu(z):
    """
    ReLU activation function.
    g(z) = max(0, z)
    """
    return np.maximum(0, z)

def _relu_grad(z):
    """
    Gradient of the ReLU function.
    g'(z) = 1 if z > 0 else 0
    """
    return np.where(z > 0, 1, 0)

def _softmax(z, axis= -1 ):
    """
    Softmax activation function.
    g(z) = exp(z) / sum(exp(z))
    """
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

