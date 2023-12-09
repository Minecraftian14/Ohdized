import numpy as np

from ohdized import ZetaNumber
from ohdized_numpy import ZetaNumpy as znp

z = lambda *p: ZetaNumber(*p)

a = z(14, 10)
print('a =', a)
b = z(12, 20, 6)
print('b =', b)

print('-a =', -a)
print('a * 0 =', a * 0)
print('a / 0 =', a / 0)
print('20 + a =', 20 + a)
print('20 * a =', 20 * a)
print('a ** 2 =', a ** 2)

print('a + b =', a + b)
print('a - b =', a - b)
print('a * b =', a * b)

x = znp.random((4, 6))
y = znp.random((6, 1))
print('x @ y =', x @ y)
print('sum y', np.sum(y))
# print('mean y', np.mean(y))
print('mean y', np.sum(y) / len(y))

print('y * 0=', y * 0)
print('1 / (y * 0) =', 1 / (y * 0))

u = ZetaNumber(1, 0, 3)
print('u =', u)
print('1 / u =', 1 / u)

v = ZetaNumber(0, 0, 2, 3, 4, 5)
print('v = ', v)
print('v * v =', v * v)
print('v^-1 =', v.multiplicative_inverse().intercept(4))
print('v^-1^-1 =', v.multiplicative_inverse().multiplicative_inverse().intercept(6))

print('', z(2, 3, 5, 7).dot(z(0, 1, 0, 1)))
