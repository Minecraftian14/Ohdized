from numbers import Number
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
        if dtype is None:
            dtype = type(args[0])
        self.value = tuple(args) if _ig else tuple(dtype(x) for x in args)
        self.value = UnboundedIterable(self.value, dtype)
        self.dtype = dtype

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
        return self.scalar_operation(other, lambda a, b: a + b)

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

    def __mul__(self, other):
        if isinstance(other, ZetaNumber):
            return self.vector_operation(other, lambda a, b: a * b)
        elif isinstance(other, Number):
            return self.scalar_operation(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self.scalar_operation(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return self.scalar_operation(other, lambda a, b: b / a)

    def __pow__(self, power, modulo=None):
        if isinstance(power, Number):
            return self.scalar_operation(power, lambda a, b: a.__pow__(power, modulo))

    def __repr__(self):
        return '+'.join([f'{x}z^{i}' for i, x in enumerate(self.value)])

    def exp(self):
        return self.unary_operation(np.exp)

    def real(self):
        return self.value[0]

    def shrink(self):
        zero = self.dtype(0)
        lift = 0
        for x in reversed(self.value):
            if x != zero: break
            lift += 1
        if lift > 0:
            self.value = self.value[:len(self.value) - lift]
            self.value = UnboundedIterable(self.value, self.dtype)
        return self

    def intercept(self, limit=10):
        if len(self.value) < limit:
            return self
        self.value = [self.value[i] for i in range(limit)]
        self.value = UnboundedIterable(self.value, self.dtype)
        return self
