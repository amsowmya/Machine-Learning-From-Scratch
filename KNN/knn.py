import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x2 - x1)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k 

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y 

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
        print("distance: ", distance)

        # get k nearest samples, labels
        k_indices = np.argsort(distance)[:self.k]
        print("indices: ", k_indices)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print("k_nearest_labels: ", k_nearest_labels)

        # majority vote, most common class label
        print(Counter(k_nearest_labels))
        most_common = Counter(k_nearest_labels).most_common(1)
        print("most_common: ", most_common)
        return most_common[0][0]
    
    # def _predict(self, x):
    #     distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
    #     indices = np.argsort(distance)[:self.k]
    #     k_nearest_labels = [self.y_train[i] for i in indices]
    #     most_common = Counter(k_nearest_labels).most_common(1)
    #     return most_common