import numpy as np

class CrossEntropy:
    """
    Cross-Entropy loss function
    L(y, y_hat) = -sum(y * log(y_hat))
    where y is the true label and y_hat is the predicted probability.
    """

    def __init__(self, weights = 1, epsilon = 1e-15):
        self.weights = weights
        self.epsilon = epsilon

    def __call__(self, y, y_hat):
        """
        Compute the cross-entropy loss.
        y : true labels (one-hot encoded), shape = (num_data, num_classes)
        y_hat : predicted probabilities, shape = (num_data, num_classes)        
        """

        assert y.shape == y_hat.shape, "Shapes of y and y_hat must match."
        y_hat[y_hat == 0] = self.epsilon  # Avoid log(0)

        loss = -np.sum(y * np.log(y_hat + self.epsilon), axis= -1)  # sum over classes
        loss = np.mean(loss)
        return loss
    

    def backward(self, y, y_hat):
        """
        Compute the gradient of the loss with respect to y_hat.
        
        dl/dz = (y_hat - y) / m
        where m is the number of samples.
        z is the input to the softmax function.
        y : true labels (one-hot encoded), shape = (num_data, num_classes)
        y_hat : predicted probabilities, shape = (num_data, num_classes)

        """        
        assert y.shape == y_hat.shape, "Shapes of y and y_hat must match."
        
        m = y.shape[0]  # number of samples

        return (y_hat - y) / m  

