import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from logistic_regression import LogisticRegression


df = datasets.load_breast_cancer()
print(df.keys())
X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
predictions = logistic_reg.predict(X_test)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc}")