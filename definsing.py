import numpy as np

from ohdized import ZetaNumber
from ohdized_numpy import ZetaNumpy as znp

z = lambda *p: ZetaNumber(*p)

a = z(2, 3)
print('a =', a)
b = z(4, 5)
print('b =', b)

print('a + b =', a + b)
print('a - b =', a - b)
print('a * b =', a * b)

print('20 + a =', 20 + a)
print('20 * a =', 20 * a)
print('a ** 2 =', a ** 2)

x = znp.random((4, 6))
y = znp.random((6, 1))
print('x @ y =', x @ y)
print('sum y', np.sum(y))
# print('mean y', np.mean(y))
print('mean y', np.sum(y) / len(y))
