import sys, os
sys.path.append(os.path.join(os.curdir, 'common'))

import numpy as np


def linear(x):
    return x


def unit_step(x):
    out = np.zeros(shape=x.shape)
    out[x >= 0] = 1
    return out


def sigmoid(x, C=250):
    return 1. / (1. + np.exp(-np.clip(x, -C, C)))


def relu(x):
    return np.maximum(0, x)


def softmax(x, C=250):
    if x.ndim == 2:
        x = x.T
        y = np.exp(-np.clip(x, -C, C)) / np.sum(np.exp(-np.clip(x, -C, C)), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
