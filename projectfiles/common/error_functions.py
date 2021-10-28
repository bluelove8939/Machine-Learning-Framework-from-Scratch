import sys, os
sys.path.append(os.path.join(os.curdir, 'common'))

import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((t - y) ** 2)


def cross_entropy_error(y, t, eta=1e-7):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + eta)) / batch_size


loss_functions = {
    'mse': mean_squared_error,
    'cee': cross_entropy_error
}