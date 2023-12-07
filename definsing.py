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
