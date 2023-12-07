from ohdized_numpy import ZetaNumpy as znp, ZetaNumpy
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, features, classes):
        self.weights = znp.random((features, classes))
        self.bias = znp.random((1, classes))

    def sigmoid(self, x):
        exp = np.exp(-x)
        return 1 / (1 + exp)

    def forward(self, x):
        g = x @ self.weights + self.bias
        a = self.sigmoid(g)
        return a, x

    def loss(self, y, yp):
        return np.sum((y - yp) ** 2) / len(y)

    def backward(self, dl, yp, cache):
        x = cache
        d = dl * yp * (1 - yp)
        dw = x.T @ d
        db = np.sum(d, axis=0, keepdims=True)
        da = d @ self.weights.T
        return da, (dw, db)

    def update(self, grads, alpha):
        dw, db = grads
        self.weights -= alpha * dw
        self.bias -= alpha * db


if __name__ == '__main__':
    x = np.asarray([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ], dtype=float)
    y = np.asarray([
        [1],
        [0],
        [0],
        [0],
    ], dtype=float)
    m = Perceptron(2, 1)
    print([u.real() for u in m.forward(x)[0].squeeze()])
    l = []
    for i in range(1000):
        yp, cache = m.forward(x)
        l.append(m.loss(y, yp).real())
        _, grads = m.backward(-(y - yp), yp, cache)
        m.update(grads, 0.01)
        m.weights = ZetaNumpy.intercept(m.weights)
        m.bias = ZetaNumpy.intercept(m.bias)
    plt.plot(l)
    plt.show()
    print([u.real() for u in m.forward(x)[0].squeeze()])
