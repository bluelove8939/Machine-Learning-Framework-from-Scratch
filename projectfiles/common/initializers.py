import sys, os
sys.path.append(os.path.join(os.curdir, 'common'))
sys.path.append(os.path.join(os.curdir, 'projectfiles/common'))

import abc
import numpy as np


class __Initializers(metaclass=abc.ABCMeta):
    def __init__(self):
        return

    @abc.abstractmethod
    def __call__(self, shape):
        return


class RandomNormal(__Initializers):
    def __init__(self, mean, stddev, random_state=None):
        super(RandomNormal, self).__init__()

        self.mean = mean
        self.stddev = stddev
        self.random_state = random_state
        self.rgen = np.random.RandomState(seed=random_state)

    def __call__(self, shape):
        return self.rgen.normal(loc=self.mean, scale=self.stddev, size=shape)


class Zeros(__Initializers):
    def __init__(self):
        super(Zeros, self).__init__()

    def __call__(self, shape):
        return np.zeros(shape=shape)


class HeNormal(__Initializers):
    def __init__(self, fan_in, random_state=None):
        super(HeNormal, self).__init__()

        self.mean = 0.
        self.stddev = np.sqrt(2.0 / fan_in)
        self.rgen = np.random.RandomState(seed=random_state)

    def __call__(self, shape):
        return self.rgen.normal(loc=self.mean, scale=self.stddev, size=shape)


if __name__ == "__main__":
    initializer = RandomNormal(mean=0, stddev=0.01)
    print(initializer(shape=(2, 5)))