from ohdized import ZetaNumber
from ohdized_numpy import ZetaNumpy as znp, ZetaNumpy
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, features, classes):
        self.weights = znp.random((2 * features - 1, classes))
        self.bias = znp.random((1, classes))

    snap = np.vectorize(lambda x: x.unary_operation(lambda u: 1e2 if u > 1e2 else u))
    real = np.vectorize(lambda x: x.real())

    def sigmoid(self, x):
        # Similar to gradient clipping
        x = Perceptron.snap(x)
        x = Perceptron.snap(-x)

        exp = np.exp(x)
        return 1 / (1 + exp)

    def transform(self, x):
        # assert len(x.shape) == 2
        return ZetaNumpy.asarray([ZetaNumber(*u) for u in x])[:, None]

    def forward(self, x):
        z = self.transform(x)
        a = z ** 2
        g = a @ self.weights[0]
        for w in self.weights[1:]:
            a = a * 0
            g += a @ w
        g = g + self.bias
        a = self.sigmoid(g)
        return a, z

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
    np.seterr(all='raise')

    x = np.asarray([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ], dtype=np.float64)
    y = np.asarray([
        [0],
        [1],
        [1],
        [0],
    ], dtype=np.float64)

    m = Perceptron(2, 1)

    print([u.real() for u in m.forward(x)[0].squeeze()])
    l = []
    for i in range(2000):
        yp, cache = m.forward(x)
        l.append(m.loss(y, yp).real())
        _, grads = m.backward(-(y - yp), yp, cache)
        m.update(grads, 0.01)
        m.weights = ZetaNumpy.intercept(m.weights)
        m.bias = ZetaNumpy.intercept(m.bias)
    plt.plot(l)
    plt.show()
    print([u.real() for u in m.forward(x)[0].squeeze()])
