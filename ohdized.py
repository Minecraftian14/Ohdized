from numbers import Number, Integral
import numpy as np


class UnboundedIterable:
    def __init__(self, v, dtype):
        self.v = v
        self.z = dtype(0)

    def __getitem__(self, i):
        s = len(self.v)
        if isinstance(i, slice):
            return self.v.__getitem__(i)
        return self.at_index(i, s)

    def at_index(self, i, s):
        return self.z if i >= s else self.v[i]

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        return self.v.__iter__()


class ZetaNumber:
    def __init__(self, *args, dtype=None, _ig=False):
        # assert len(args) > 0
        if dtype is None: dtype = type(args[0])
        if len(args) == 0: args = [0]
        self.value = tuple(args) if _ig else tuple(dtype(x) for x in args)
        self.value = UnboundedIterable(self.value, dtype)
        self.dtype = dtype

    @staticmethod
    def __extended_binomial__(x: Number, l=10):
        """Expansion of (1+x)^-1 upto l terms"""
        if x.real() > 1: return ZetaNumber.__extended_binomial__(1 / x, l) / x
        return sum(np.power(-x, k) for k in range(0, l))

    def multiplicative_inverse(self, l=10):
        # Review needed
        zero = self.dtype(0)
        first_non_zero_term = None
        for i, x in enumerate(self.value):
            if x == zero: continue
            first_non_zero_term = i
            break
        if first_non_zero_term is None:
            raise Exception
        T = list(self.value)
        D = ZetaNumber(dtype=self.dtype, _ig=True, *T[:first_non_zero_term + 1])
        T[first_non_zero_term] = zero
        N = ZetaNumber(dtype=self.dtype, _ig=True, *T)
        x = N / D  # Hopefully causes a monomial denominator call instead of an endless recursion
        ans = ZetaNumber.__extended_binomial__(x, l) / self.value[first_non_zero_term]
        for _ in range(first_non_zero_term):
            ans = ans / 0
        return ans

    def unary_operation(self, operator):
        value = [operator(x) for x in self.value]
        return ZetaNumber(dtype=self.dtype, _ig=True, *value)

    def scalar_operation(self, other, operator):
        if isinstance(other, ZetaNumber):
            dtype = type(operator(self.value[0], other.value[0]))
            size = max(len(self.value), len(other.value))
            value = tuple(operator(self.value[i], other.value[i]) for i in range(size))
        elif isinstance(other, Number):
            dtype = type(operator(self.value[0], other))
            value = tuple(operator(i, other) for i in self.value)
        else:
            return None
        return ZetaNumber(dtype=dtype, _ig=True, *value)

    def vector_operation(self, other, operator, identity=0):
        if isinstance(other, ZetaNumber):
            dtype = type(operator(self.value[0], other.value[0]))
            length = len(self.value) + len(other.value) - 1
            zero = dtype(identity)
            value = [zero for _ in range(length)]

            for i in range(len(self.value)):
                for j in range(len(other.value)):
                    value[i + j] += operator(self.value[i], other.value[j])
        else:
            return None
        return ZetaNumber(dtype=dtype, _ig=True, *value).shrink()

    def __add__(self, other):
        if isinstance(other, ZetaNumber):
            return self.scalar_operation(other, lambda a, b: a + b)
        elif isinstance(other, Number):
            home = self.value[0] + other
            dtype = type(home)
            value = [home, ] + [dtype(u) for u in self.value[1:]]
            return ZetaNumber(dtype=dtype, _ig=False, *value)

    def __radd__(self, other):
        return self.__add__(other)

    def __pos__(self):
        # Wait what?
        return self

    def __sub__(self, other):
        return self.scalar_operation(other, lambda a, b: a - b)

    def __rsub__(self, other):
        # To be reviewed
        return self.__add__(-other)

    def __neg__(self):
        return self.unary_operation(lambda x: -x)

    def dot(self, other):
        return self.scalar_operation(other, lambda a, b: a * b)

    def __mul__(self, other):
        if isinstance(other, ZetaNumber):
            return self.vector_operation(other, lambda a, b: a * b)
        elif isinstance(other, Number):
            if self.dtype(0) == other:
                value = [self.value[u] for u in range(1, len(self.value))]
                if len(value) == 0:
                    value = (0,)
                return ZetaNumber(dtype=self.dtype, _ig=True, *value)
            else:
                return self.scalar_operation(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other, l=10):
        if isinstance(other, ZetaNumber):
            if other.number_of_non_zero_terms() == 1:
                # monomial denominator division
                for i, v in enumerate(other.value):
                    if v == 0: continue
                    if i >= len(self.value): return ZetaNumber(dtype=self.dtype)
                    return ZetaNumber(dtype=self.dtype, *self.value[i:]) / v
            else:
                return self * other.multiplicative_inverse(l)
        elif isinstance(other, Number):
            if self.dtype(0) == other:
                value = [0, ] + [u for u in self.value]
                return ZetaNumber(dtype=self.dtype, _ig=True, *value).shrink()
            else:
                return self.scalar_operation(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            if len(self.value) == 1 and self.value[0] == self.dtype(0):
                return ZetaNumber(self.dtype(other)) / 0
            else:
                # TODO: Add (poly)^-1 expansion formula
                dtype = type(other / self.dtype(1))
                value = tuple((0 if i == 0 else other / i) for i in self.value)
                return ZetaNumber(dtype=dtype, _ig=True, *value)

    def __pow__(self, power, modulo=None):
        # if isinstance(power, Number):
        #     return self.scalar_operation(power, lambda a, b: a.__pow__(power, modulo))
        if isinstance(power, Integral):
            if power == 0:  # No idea what to do actually
                if self.number_of_non_zero_terms() == 0:
                    return ZetaNumber(0, dtype=self.dtype)
                else:
                    return ZetaNumber(1, dtype=self.dtype)
            elif power == 1:
                return self
            elif power > 1:
                z = ZetaNumber(1, dtype=self.dtype)
                for _ in range(power):
                    z = z * self
                return z

    def __repr__(self):
        # return '+'.join([f'{x}z^{i}' for i, x in enumerate(self.value)])
        return 'z ' + ' '.join([f'{x:.2f}' for i, x in enumerate(self.value)])

    def __str__(self):
        return self.__repr__()

    def number_of_non_zero_terms(self):
        zero = self.dtype(0)
        return sum(0 if x == zero else 1 for x in self.value)

    def exp(self):
        return self.unary_operation(np.exp)

    def real(self):
        return self.value[0]
        # return sum(self.value)

    def shrink(self):
        zero = self.dtype(0)
        lift = 0
        for x in reversed(self.value):
            if x != zero: break
            lift += 1
        if lift > 0:
            self.value = self.value[:max(1, len(self.value) - lift)]
            self.value = UnboundedIterable(self.value, self.dtype)
        return self

    def intercept(self, limit=10):
        if len(self.value) < limit:
            return self
        self.value = [self.value[i] for i in range(limit)]
        self.value = UnboundedIterable(self.value, self.dtype)
        return self

    @staticmethod
    def __clip__(x, limit):
        if x < -limit: return -limit
        if x > limit: return -limit
        return x

    def clip(self, limit=100):
        self.value = [ZetaNumber.__clip__(v, limit) for v in self.value]
        self.value = UnboundedIterable(self.value, self.dtype)
        return self
