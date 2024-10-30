import numpy as np


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def loss(self, y, y_predicted):
        return np.mean((y_predicted - y)**2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            loss = self.loss(y, y_predicted)

            dw = 1/(n_samples) * np.dot(X.T, (y_predicted - y))
            db = 1/(n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch%100 == 0:
                print(f"Epoch: {epoch}/{self.n_iters}, Weight: {self.weights}, Loss: {loss:.4f}")

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted