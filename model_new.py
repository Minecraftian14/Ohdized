import matplotlib.pyplot as plt
import numpy as np

from ohdized import ZetaNumber
from ohdized_numpy import ZetaNumpy as znp


class Perceptron:
    def __init__(self, _dtype=np.float64, _activity=10):
        self.expansion = znp.random1(dtype=_dtype, activity=_activity)
        # self.expansion = ZetaNumber(1, 1, 1, 1, dtype=_dtype)
        self.expanding_bias = znp.random1(dtype=_dtype, activity=_activity)
        # self.expanding_bias = ZetaNumber(0, -1.5, 0, -.5, 0, dtype=_dtype)
        self.contraction = znp.random1(dtype=_dtype, activity=_activity)
        # self.contraction = ZetaNumber(0, -1, 0, 1, dtype=_dtype)
        self.contracting_bias = znp.random1(dtype=_dtype, activity=_activity)
        # self.contracting_bias = ZetaNumber(0, -0.22, dtype=_dtype)
        self.activity = _activity

    @staticmethod
    def features_to_zetas(_x):
        return znp.asarray([ZetaNumber(*sample) for sample in _x])[:, None]

    snapUp = np.vectorize(lambda x: x.unary_operation(lambda u: 1e2 if u > 1e2 else u))
    snapDown = np.vectorize(lambda x: x.unary_operation(lambda u: -1e2 if u < -1e2 else u))
    dot = np.vectorize(lambda x, y: x.dot(y))
    scalar_inverse = np.vectorize(lambda x: x.unary_operation(lambda u: 1 / u if u != 0 else 0))
    scalar_unify = np.vectorize(lambda x: x.unary_operation(lambda u: u + 1))

    @staticmethod
    def activate(_x):
        _x = Perceptron.snapUp(_x)
        _x = Perceptron.snapDown(_x)
        # act = 1 / (1 + np.exp(-_x))
        act = Perceptron.scalar_inverse(Perceptron.scalar_unify(np.exp(-_x)))
        de_act = act * (1 - act)
        return act, de_act

    @staticmethod
    def loss(y, yp):
        loss = (y - yp) ** 2 / 2
        de_loss = (y - yp)
        return loss, de_loss

    @staticmethod
    def present(yp):
        return [u.real() for u in yp.squeeze()]

    val_su = np.vectorize(lambda u: np.sum(u.value))

    def forward(self, _x):
        z = Perceptron.features_to_zetas(_x)
        g = z * self.expansion + self.expanding_bias
        # g = Perceptron.dot(z, self.expansion) + self.expanding_bias
        a, ad = Perceptron.activate(g)
        # h = a / self.contraction + self.contracting_bias
        # h = a * self.contraction + self.contracting_bias
        h = Perceptron.dot(a, self.contraction) + self.contracting_bias
        b = Perceptron.val_su(znp.intercept(h, self.activity))
        bd = 1
        # b, bd = Perceptron.activate(znp.intercept(h, self.activity))
        # b = Perceptron.val_su(b)
        return b, (z, a, ad, bd)

    def backward(self, _dl, _yp, cache):
        z, a, ad, bd = cache
        db = _dl
        # dh = db * bd
        # dh = Perceptron.dot(db, bd)
        dh = db
        # dconw = -dh * a / (self.contraction ** 2)
        # dconw = dh * a
        dconw = Perceptron.dot(a, dh)
        dconb = dh
        # da = dh / self.contraction
        # da = dh * self.contraction
        da = Perceptron.dot(self.contraction, dh)
        # dg = da * ad
        dg = Perceptron.dot(da, ad)
        # dexpw = dg * z
        dexpw = Perceptron.dot(dg, z)
        dexpb = dg
        dz = Perceptron.dot(dg, self.expansion)
        return dz, (dconw, dconb, dexpw, dexpb)

    def update(self, grads, alpha):
        dconw, dconb, dexpw, dexpb = grads
        self.expansion = self.expansion + alpha * np.sum(dexpw)
        self.expanding_bias = self.expanding_bias + alpha * np.sum(dexpb)
        self.contraction = self.contraction + alpha * np.sum(dconw)
        self.contracting_bias = self.contracting_bias + alpha * np.sum(dconb)
        self.expansion.intercept(self.activity).clip()
        self.expanding_bias.intercept(self.activity).clip()
        self.contraction.intercept(self.activity).clip()
        self.contracting_bias.intercept(self.activity).clip()

    def train(self, _x, _y, _alpha):
        yp, cache = self.forward(_x)
        loss, dl = Perceptron.loss(_y, yp)
        _, grads = self.backward(dl, yp, cache)
        self.update(grads, _alpha)
        return loss


if __name__ == '__main__':
    np.seterr(all='raise')

    x = np.asarray([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ])
    y = np.asarray([
        [1],
        [0],
        [0],
        [0],
    ])
    m = Perceptron()
    print(m.forward(x)[0])
    # print(Perceptron.present(m.forward(x)[0]))
    ls = []
    for i in range(200):
        if i % 100 == 0:
            print(i)
        l = m.train(x, y, 0.01)
        ls.append(np.sum(l))
    print(m.forward(x)[0])
    # print(Perceptron.present(m.forward(x)[0]))
    plt.plot(ls)
    plt.show()
