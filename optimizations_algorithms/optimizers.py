import numpy as np

class _Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def minimize(self, gradient: np.ndarray) -> np.ndarray:
        """
        Minimize the loss function using the optimizer.
        gradient: ndarray, gradient of the loss function
        Returns: ndarray, updated weights
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    

class SGD(_Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def minimize(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update the weights using SGD.
        gradient: ndarray, gradient of the loss function
        Returns: ndarray, updated weights
        """
        return self.learning_rate * gradient
    
    