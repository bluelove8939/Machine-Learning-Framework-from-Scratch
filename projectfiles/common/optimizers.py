import sys, os
sys.path.append(os.path.join(os.curdir, 'common'))

import abc
import numpy as np


class __Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        return

    @abc.abstractmethod
    def __call__(self, params, grads):
        return self


class SGD(__Optimizer):
    def __init__(self, lr=0.01, momentum=0., decay=0.):
        super(SGD, self).__init__()

        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.velocity = None

    def __call__(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros(shape=grad.shape) for grad in grads]

        if self.lr > 0:
            self.lr *= (1 - self.decay)

        for idx, grad in enumerate(grads):
            self.velocity[idx] = self.momentum * self.velocity[idx] - self.lr * grad
            params[idx] += self.velocity[idx]

        return self


class AdaGrad(__Optimizer):
    def __init__(self, lr, epsilon=1e-8):
        super(AdaGrad, self).__init__()

        self.lr = lr
        self.epsilon = epsilon
        self.grad_squared = None
        self.velocity = None

    def __call__(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros(shape=grad.shape) for grad in grads]

        if self.grad_squared is None:
            self.grad_squared = [grad ** 2 for grad in grads]
        else:
            for idx in range(len(self.grad_squared)):
                self.grad_squared[idx] += grads[idx] ** 2

        for idx, grad in enumerate(grads):
            self.velocity[idx] = - self.lr * grad / (np.sqrt(self.grad_squared[idx] + self.epsilon))
            params[idx] += self.velocity[idx]

        return self


class Adam(__Optimizer):
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(Adam, self).__init__()

        self.lr = lr
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first_momentum = None
        self.second_momentum = None
        self.iter = 0

    def __call__(self, params, grads):
        if self.first_momentum is None:
            self.first_momentum = [np.zeros(shape=grad.shape) for grad in grads]

        if self.second_momentum is None:
            self.second_momentum = [np.zeros(shape=grad.shape) for grad in grads]

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta_2 ** self.iter) / (1.0 - self.beta_1 ** self.iter)

        for idx, grad in enumerate(grads):
            self.first_momentum[idx] = self.beta_1 * self.first_momentum[idx] + (1 - self.beta_1) * grad
            self.second_momentum[idx] = self.beta_2 * self.second_momentum[idx] + (1 - self.beta_2) * grad ** 2

            params[idx] -= lr_t * self.first_momentum[idx] / (np.sqrt(self.second_momentum[idx]) + self.epsilon)

        return self


optimizer_units = {
    'sgd': SGD,
    'adagrad': AdaGrad
}

default_params = {
    'sgd': (0.01, 0., 0.),
    'adagrad': (0.01, 1e-8)
}