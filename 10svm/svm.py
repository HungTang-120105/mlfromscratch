import numpy as np
from scipy.spatial.distance import cdist
from cvxopt import solvers
import cvxopt

class SVM:

    kernels = {
        'linear': '_linear_kernel',
        'poly': '_polynomial_kernel',
        'rbf': '_rbf_kernel',
        'sigmoid': '_sigmoid_kernel'
    }

    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', debug = False,  is_saved = False):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.debug = debug
        self.is_saved = is_saved
        self.supoprt_vectors_ = None
        self._alpha = None
        self.w = None
        self.b = None

    def _linear_kernel(self, X1, X2):
        # k(X1, X2) = X1 * X2.T
        return np.dot(X1, X2.T)
    
    def _polynomial_kernel(self, X1, X2):
        # k(X1, X2) = (X1 * X2.T + 1) ** degree
        return (np.dot(X1, X2.T) + 1) ** self.degree
    
    def _rbf_kernel(self, X1, X2):
        # k(X1, X2) = exp(-gamma * ||X1 - X2||^2)
        if self.gamma == 'auto':
            self.gamma = 1.0 / X1.shape[1]
        return np.exp(-self.gamma * cdist(X1, X2, 'sqeuclidean'))
    
    def _sigmoid_kernel(self, X1, X2):
        # k(X1, X2) = tanh(X1 * X2.T + 1)
        return np.tanh(self._linear_kernel(X1, X2) + 1)

    def _solve_lagrange_dual_problem(self, X, y):
        """
        use cvxopt to solve the Lagrange dual problem
        min 1/2 x.T * P * x + q.T * x
        s.t. G * x <= h
             A * x = b
        where x is the Lagrange multipliers 

        our problem is:
        min 1/2 * sum_i(sum_j(y_i * y_j * alpha_i * alpha_j * k(X_i, X_j))) - sum(alpha_i)
        s.t. sum(y_i * alpha_i) = 0
             0 <= alpha_i <= C
        """
        # shape(alpha_i) = (N, 1)
        N, d = X.shape
        y = np.reshape(y, (N, 1)).astype(np.double) # shape(y) = (N, 1)
        
        _func = getattr(self, self.kernels[self.kernel])
        K = _func(X, X)
        K = K * (y @ y.T)

        P = cvxopt.matrix(K)

        q = cvxopt.matrix(-np.ones((N, 1)))

        G = cvxopt.matrix(np.vstack((np.eye(N) * -1, np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
        
        A = cvxopt.matrix(y.T)  # (1, N), kiểu double
        b = cvxopt.matrix(np.zeros(1))      # (1, 1)

        solvers.options['show_progress'] = self.debug
        sol = solvers.qp(P, q, G, h, A, b)

        alpha = np.array(sol['x']).flatten()
        if self.is_saved:
            np.save('alpha.npy', alpha)
        return alpha
    
    def sol_svm(self, X, y):
        """
        solve SVM problem
        """
        # solve Lagrange dual problem
        alpha = self._solve_lagrange_dual_problem(X, y)

        # get support vectors
        sv = np.where(alpha > 1e-5)[0]
        self.supoprt_vectors_ = X[sv]

        on_margin = np.where((alpha > 1e-5) & (alpha < self.C))[0]

        # get weights and bias
        # w = sigma(y_i * alpha_i * X_i) ( i in S: support vectors)
        # b = average(y_i - w * X_i) ( i in M: vectors on margin)
        self.w = np.sum((alpha[sv] * y[sv])[:, np.newaxis] * self.supoprt_vectors_, axis=0)
        self.b = np.mean(y[on_margin] - np.dot(self.w, X[on_margin].T))
        if self.is_saved:
            np.save('w.npy', self.w)
            np.save('b.npy', self.b)

        return self.w, self.b
    
    def fit(self, X, y):
        assert len(np.unique(y)) == 2, "y must have only 2 unique values"
        assert self.kernel in self.kernels, "kernel must be one of {}".format(list(self.kernels.keys()))
        
        # convert y to -1 and 1
        y = np.where(y == 0, -1, 1)

        # solve SVM problem
        self.sol_svm(X, y)
        if self.is_saved:
            np.save('sv.npy', self.supoprt_vectors_)       

    def predict(self, X_new):
        pred = np.dot(X_new, self.w) + self.b
        pred = np.sign(pred)
        pred = np.where(pred == -1, 0, 1)

        return pred
    




        



    

    