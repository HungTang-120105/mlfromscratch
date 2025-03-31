import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import io
import imageio

class LinearRegression:
    def __init__(self, alpha, epochs, lambda_ = 0.1, do_visualize = False):
        self.alpha = alpha
        self.epochs = epochs
        self.lambda_ = lambda_
        self.do_visualize = do_visualize
        self.w = None
        self.b = None
        self.vis_elems ={
            "loss": [],
            "iteration": [],
            "weights": [],
            "bias": [],
        }
    
    def _hypothesis(self, X):
        return np.dot(X, self.w) + self.b
    
    # Ridge regression cost function
    def mse_loss(self, X, y, y_hat):
        m = X.shape[0]
        return 1/(2*m)*np.sum((y - y_hat) ** 2) + 1/(2*m)*self.lambda_*np.linalg.norm(self.w, ord=2)**2
    
    # Gradient descent for Ridge regression
    # dloss/dw = 1/m * X.T * (y_hat - y) + 1/m * lambda_ * w
    def _gradient(self, X, y, y_hat):
        m = X.shape[0]
        return 1/m * np.dot(X.T, (y_hat - y)) + 1/m * self.lambda_ * self.w
    
    # dloss/db = 1/m * sum(y_hat - y)
    def _gradient_bias(self, y, y_hat):
        m = y.shape[0]
        return 1/m * np.sum(y_hat - y)
    
    def _train_one_epoch(self, X_train, y_train, e):
        # compute the hypothesis
        y_hat = self._hypothesis(X_train)

        # compute the loss
        loss = self.mse_loss(X_train, y_train, y_hat)
        print(f"Epoch {e}/{self.epochs}, Loss: {loss:.4f}")

        # compute the gradients
        dw = self._gradient(X_train, y_train, y_hat)
        db = self._gradient_bias(y_train, y_hat)

        # update the weights and bias
        self.w -= self.alpha * dw
        self.b -= self.alpha * db

        w_grad_norm = np.linalg.norm(dw, ord=2)

        return loss, w_grad_norm
    
    def train(self, X_train, y_train):
        self.w = np.random.normal(size=(X_train.shape[1], 1))
        self.b = np.random.normal(size=(1, 1))

        for e in range(self.epochs):
            Loss, w_grad_norm = self._train_one_epoch(X_train, y_train, e)
            
            if self.do_visualize and (e+1) % 5 == 0:
                self.vis_elems["loss"].append(Loss)
                self.vis_elems["iteration"].append(e+1)
                self.vis_elems["weights"].append(copy.deepcopy(self.w))
                self.vis_elems["bias"].append(copy.deepcopy(self.b))

            if w_grad_norm < 1e-6:
                break

    def _plot(self, w, b, loss, iteration, X, X_transform, y):
        y_plot = self._hypothesis(X_transform, w, b)
        plt.figure(0, figsize=(6, 6))
        plt.clf()
        plt.title("Loss: " + str(loss))
        plt.scatter(X[:, 0], y, color='r')
        label = "Iteration: " + str(iteration)
        for ind, t in enumerate(loss):
            label += "\nTheta %s: %.2f" % (ind, t)
        plt.plot(X, y_plot, '-', label=label)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        return imageio.imread(img_buf)

    def create_gif(self, X, X_transform, y):
        imgs = []
        for l, i, w, b in zip(self.vis_elems["loss"], self.vis_elems["iteration"], self.vis_elems["weight"], self.vis_elems["bias"]):
            imgs.append(self._plot(w, b, l, i, X, X_transform, y))
        imageio.mimsave("linear_regressionss.gif", imgs, fps=5)


    def predict(self, X_test):
        return self._hypothesis(X_test)
    
    # r2_score = 1 - (SS_res / SS_tot)
    # SS_res = sum((y - y_hat) ** 2)
    # SS_tot = sum((y - y_mean) ** 2)
    def r2_score(self, y_hat, y_test):
        SS_res = np.sum((y_test - y_hat) ** 2)
        SS_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (SS_res / SS_tot)

