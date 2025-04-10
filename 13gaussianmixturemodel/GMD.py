import numpy as np

class GMD:
    def __init__(self, n_components=1, max_iter=100, tol= 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def multivariate_normal(self, X, mean, cov):
        cov += np.eye(cov.shape[0]) * 1e-6  # Add small value to covariance matrix for numerical stability
        d = X.shape[1]
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        coeff = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(cov) ** 0.5)
        return coeff * np.exp(exponent)
    

    def fit(self, X):
        """
        Input:
    - Dữ liệu X = {x₁, x₂, ..., x_N}, mỗi xₙ ∈ ℝ^d
    - Số lượng thành phần Gaussian: K
    - Số vòng lặp tối đa: max_iter
    - Ngưỡng hội tụ: ε

    Khởi tạo:
        - Trọng số π_i ~ Uniform (sao cho ∑ π_i = 1)
        - Trung bình μ_i ~ Random chọn từ X
        - Ma trận hiệp phương sai Σ_i = identity matrix (hoặc sample covariance)

    For iter = 1 to max_iter:
        
        --- E-step ---
        For mỗi điểm dữ liệu xₙ:
            For mỗi thành phần Gaussian i:
                Tính γ_i(xₙ) = trách nhiệm của Gaussian i cho xₙ:
                    γ_i(xₙ) = π_i * N(xₙ | μ_i, Σ_i)
            Normalize:
                γ_i(xₙ) = γ_i(xₙ) / ∑_{j=1}^{K} γ_j(xₙ)

        --- M-step ---
        For mỗi Gaussian i:
            N_i = ∑_{n=1}^{N} γ_i(xₙ)          (Tổng trách nhiệm)

            Cập nhật trung bình:
                μ_i = (1 / N_i) * ∑_{n=1}^{N} γ_i(xₙ) * xₙ

            Cập nhật hiệp phương sai:
                Σ_i = (1 / N_i) * ∑_{n=1}^{N} γ_i(xₙ) * (xₙ - μ_i)(xₙ - μ_i)^T

            Cập nhật trọng số:
                π_i = N_i / N

        --- Kiểm tra hội tụ ---
        Tính log-likelihood L:
            L = ∑_{n=1}^{N} log ( ∑_{i=1}^{K} π_i * N(xₙ | μ_i, Σ_i) )

        Nếu |L - L_prev| < ε:
            Break

        L_prev ← L
    Output:
        - Các tham số {π_i, μ_i, Σ_i} cho mỗi Gaussian
        - Ma trận trách nhiệm γ (dùng để phân cụm hoặc dự đoán xác suất)
        """

        # Initialize parameters
        self.n_samples, self.n_features = X.shape
        self.pi = np.ones(self.n_components) / self.n_components  # Mixing coefficients
        self.mu = X[np.random.choice(self.n_samples, self.n_components, replace=False)]
        self.sigma = np.array([np.eye(self.n_features)] * self.n_components) # shape = (K, d, d)
        self.gamma = np.zeros((self.n_samples, self.n_components))
        self.log_likelihood = 0

        for iter in range(self.max_iter):

            # E-step: Calculate responsibilities
            for k in range(self.n_components):
                self.gamma[:, k] = self.pi[k] * self.multivariate_normal(X, self.mu[k], self.sigma[k])
            self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)

            # M-step: Update parameters
            for k in range(self.n_components):
                N_k = np.sum(self.gamma[:, k])
                self.mu[k] = np.sum(self.gamma[:, k].reshape(-1, 1) * X, axis=0) / N_k
                diff = X - self.mu[k]
                self.sigma[k] = np.dot((self.gamma[:, k] * diff.T), diff) / N_k
                self.pi[k] = N_k / self.n_samples

            # Calculate log-likelihood
            log_likelihood_new = np.sum(np.log(np.sum(self.gamma, axis=1)))
            if np.abs(log_likelihood_new - self.log_likelihood) < self.tol:
                break
            self.log_likelihood = log_likelihood_new

    def predict(self, X):
        """
        Predict the cluster for each sample in X.
        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Data to predict.
        Returns:
        - labels: array, shape (n_samples,)
            Predicted cluster for each sample.
        """
        self.gamma = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            self.gamma[:, k] = self.pi[k] * self.multivariate_normal(X, self.mu[k], self.sigma[k])
        return np.argmax(self.gamma, axis=1)
    
    


