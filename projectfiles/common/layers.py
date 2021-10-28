import sys, os
sys.path.append(os.path.join(os.curdir, 'common'))

import numpy as np
import abc

import initializers
from activation_functions import sigmoid, relu, softmax
from error_functions import cross_entropy_error, mean_squared_error


class __ConnectionLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass


class __ActivationLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass


class __LossLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, y, t):
        pass

    @abc.abstractmethod
    def backward(self, dout=1):
        pass


class __NetworkLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.params = []  # parameters of the network layer
        self.grads = []   # gradient of each parameters

    @abc.abstractmethod
    def compile(self):
        return self

    @abc.abstractmethod
    def forward(self, x):
        return x

    @abc.abstractmethod
    def backward(self, dout):
        return dout


class Affine(__ConnectionLayer):
    def __init__(self, w, b):
        super(Affine, self).__init__()
        self.w = w  # weight vector
        self.b = b  # bias value

        self.x = None
        self.dw = None  # gradient of weight vector
        self.db = None  # gradient of bias value

    def forward(self, x):
        self.x = x

        return np.dot(self.x, self.w) + self.b

    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return np.dot(dout, self.w.T)


class Linear(__ActivationLayer):
    def __init__(self):
        super(Linear, self).__init__()
        self.out = None

    def forward(self, x):
        self.out = x

        return self.out

    def backward(self, dout):
        return dout


class Sigmoid(__ActivationLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)

        return self.out

    def backward(self, dout):
        return dout * (1. - self.out) * self.out


class Relu(__ActivationLayer):
    def __init__(self):
        super(Relu, self).__init__()
        self.out = None

    def forward(self, x):
        self.out = relu(x)

        return self.out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.out == 0] = 0

        return dx


class Softmax(__ActivationLayer):
    def __init__(self):
        super(Softmax, self).__init__()
        self.out = None

    def forward(self, x):
        self.out = softmax(x)

        return self.out

    def backward(self, dout):
        return dout * self.out * (1. - self.out)


class CrossEntropyError(__LossLayer):
    def __init__(self):
        super(CrossEntropyError, self).__init__()
        self.out = None
        self.y = None
        self.t = None

    def forward(self, y, t, eta=1e-07):
        self.y = y
        self.t = t
        self.out = cross_entropy_error(y, t, eta=eta)

        return self.out

    def backward(self, dout=1):
        return dout * (self.t - self.y)


class MeanSquaredError(__LossLayer):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.out = None
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y = y
        self.t = t
        self.out = mean_squared_error(y, t)

        return self.out

    def backward(self, dout=1):
        return dout * (self.t - self.y)


class Dense(__NetworkLayer):
    def __init__(self, units, activation='linear', input_shape=None,
                 kernel_initializer=initializers.RandomNormal(0., 0.01),
                 bias_initializer=initializers.Zeros()):

        super(Dense, self).__init__()

        if not isinstance(units, int):
            raise TypeError("'units' needs to be an integer")

        if isinstance(activation, str):
            if activation not in activation_units.keys():
                raise TypeError(f"activation type '{activation}' not found")
        else:
            raise TypeError("'activation' needs to be a string")

        self.units              = units               # output size
        self.activation_type    = activation.lower()  # activation type
        self.input_shape        = input_shape         # input size preset
        self.kernel_initializer = kernel_initializer  # kernel initializer
        self.bias_initializer   = bias_initializer    # bias initializer

        # parameters and gradients
        self.params  = [None, None]  # first element:  kernel, gradient of kernel
        self.grads   = [None, None]  # second element: bias, gradient of bias

        # sublayers
        self.connection = None  # layer for affine transform
        self.activation = None  # layer for activation

    def compile(self, random_state=None):
        if self.input_shape is None:
            raise TypeError(f"'input_shape' not defined")

        # initializing parameters
        self.params[0] = self.kernel_initializer(shape=(self.input_shape, self.units))
        self.params[1] = self.bias_initializer(shape=self.units)

        # generating sublayers
        self.connection = Affine(w=self.params[0], b=self.params[1])
        self.activation = activation_units[self.activation_type]()

        return self

    def forward(self, x):
        connection_out = self.connection.forward(x)
        activation_out = self.activation.forward(connection_out)

        return activation_out

    def backward(self, dout):
        activation_dout = self.activation.backward(dout)
        connection_dout = self.connection.backward(activation_dout)

        self.grads[0] = self.connection.dw
        self.grads[1] = self.connection.db

        return connection_dout


class Flatten(__NetworkLayer):
    def __init__(self, input_shape=None):
        super(Flatten, self).__init__()

        if not isinstance(input_shape, tuple):
            raise TypeError("'input_shape' needs to be a tuple")

        self.units       = np.multiply.reduce(input_shape)
        self.input_shape = input_shape  # input size preset
        self.batch_size  = None         # batch size

    def compile(self, random_state=None):
        if self.input_shape is None:
            raise TypeError(f"'input_shape' not defined")

        return self

    def forward(self, x):
        out = np.reshape(x, newshape=(x.shape[0], self.units))
        self.batch_size = x.shape[0]

        return out

    def backward(self, dout):
        dx = np.reshape(dout, newshape=(self.batch_size, *self.input_shape))

        return dx


activation_units = {
    'linear' : Linear,
    'sigmoid': Sigmoid,
    'relu'   : Relu,
    'softmax': Softmax
}


loss_units = {
    'cee': CrossEntropyError,
    'categorical_crossentropy': CrossEntropyError,
    'mse': MeanSquaredError,
    'mean_squared': MeanSquaredError
}
