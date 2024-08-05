#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 3, 2, 3, 5])

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, predictions, color='red', label='Fitted line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Simple Linear Regression Example')
    plt.legend()
    plt.show()

    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}')

if __name__ == '__main__':
    main()
