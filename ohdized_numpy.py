import numpy as np

from ohdized import ZetaNumber


class ZetaNumpy:
    intercept = np.vectorize(lambda x, v=10: x.intercept(v))
    upgrade = np.vectorize(lambda x: ZetaNumber(x))

    @staticmethod
    def asarray(x):
        return np.asarray(x, dtype=np.object_)

    @staticmethod
    def random(size, dtype=None, decay=0.3):
        if size is None:
            return ZetaNumpy.asarray(ZetaNumpy.random1(dtype, decay))
        else:
            amount = np.prod(size)
            values = ZetaNumpy.randomN(amount, dtype, decay)
            return ZetaNumpy.asarray(values).reshape(size)
        pass

    @staticmethod
    def random1(dtype=None, decay=0.3, activity=None):
        activity = activity if activity is not None else int(1 + np.log(np.random.random()) / np.log(decay))
        value = (np.random.random() for _ in range(activity))
        return ZetaNumber(dtype=dtype, _ig=True, *value)

    @staticmethod
    def randomN(size, dtype=None, decay=0.3):
        return [ZetaNumpy.random1(dtype, decay) for _ in range(size)]
