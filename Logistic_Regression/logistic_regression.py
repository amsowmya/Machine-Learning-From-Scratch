import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter 
        self.weights = None
        self.bias = None 

    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            loss = self.loss(y, y_predicted)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db 

            if (epoch % 100) == 0:
                print(f"Epoch: {epoch}/{self.n_iter}, Weights: {self.weights}, Loss: {loss}")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def loss(self, y, y_pred):
        return np.mean((y_pred - y)**2)