import numpy as np
import matplotlib.pyplot as plt



def linear_regression_one_dimension():
    X = []
    Y = []
    with open('linear_regression/data_1d.csv') as fp:
        for line in fp.readlines():
            x, y = line.split(',')
            X.append(float(x))
            Y.append(float(y))
    X = np.array(X)
    Y = np.array(Y)

    plt.scatter(X, Y)
    plt.show()

    denominator = X.dot(X) - X.mean() * X.sum()
    a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
    b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

    Yhat = a*X + b
    plt.scatter(X, Y)
    plt.plot(X, Yhat)
    plt.show()

    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - (d1.dot(d1) / d2.dot(d2))
    print(f"R Squared: {r2}")


if __name__ == '__main__':
    linear_regression_one_dimension()
