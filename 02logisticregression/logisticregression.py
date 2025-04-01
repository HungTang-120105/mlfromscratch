import numpy as np


class LogisticRegression:
    def __init__(self, epochs, learning_rate=0.01, batch_size=32):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.w = None
        self.vis_elems = {
            "loss": [],
            "iteration": [],
            "weights": [],
        }

    def _hypothesis(self, X):
        return np.dot(X, self.w) 
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _cross_entropy_loss(self, y, y_hat):
        return -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
    
    def _gradient(self, X, y, y_hat):
        m = X.shape[0]
        return 1/m * np.dot(X.T, (y_hat - y))
    
    def _train(self, X_train, y_train):
        m = X_train.shape[0]
        indices = np.arange(m)
        np.random.shuffle(indices)

        for e in range(self.epochs):
            batch_loss = 0
            num_batches = 0
            iteration = 0

            while(iteration < X_train.shape[0]):
                batch_indices = indices[iteration:iteration + self.batch_size]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                y_hat = self._sigmoid(self._hypothesis(X_batch))
                loss = self._cross_entropy_loss(y_batch, y_hat)
                batch_loss += loss
                num_batches += 1

                dw = self._gradient(X_batch, y_batch, y_hat)
                self.w -= self.learning_rate * dw

                iteration += self.batch_size

                self.vis_elems["loss"].append(loss)
                self.vis_elems["iteration"].append(e * num_batches + iteration // self.batch_size)
                self.vis_elems["weights"].append(self.w.copy())

            print(f"Epoch {e}/{self.epochs}, Loss: {batch_loss / num_batches:.4f}")

    def train(self, X_train, y_train):
        assert type(X_train) == np.ndarray, "X_train must be a numpy array"
        assert type(y_train) == np.ndarray, "y_train must be a numpy array"

        y_train = y_train.reshape(-1, 1)

        self.w = np.random.normal(size=(X_train.shape[1], 1))

        self._train(X_train, y_train)

    def predict(self, X_test):
        assert type(X_test) == np.ndarray, "X_test must be a numpy array"

        y_hat = self._sigmoid(self._hypothesis(X_test))
        return np.where(y_hat >= 0.5, 1, 0)
    
    def accuracy(self, y_pred, y_test):

        assert type(y_pred) == np.ndarray, "y_pred must be a numpy array"
        assert type(y_test) == np.ndarray, "y_test must be a numpy array"

        y_pred = y_pred.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return np.mean(y_pred == y_test)
    
    def r2_score(self, y_pred, y_test):
        assert type(y_pred) == np.ndarray, "y_pred must be a numpy array"
        assert type(y_test) == np.ndarray, "y_test must be a numpy array"

        y_pred = y_pred.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)

        return 1 - (ss_residual / ss_total)

        