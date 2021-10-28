import sys, os
sys.path.append(os.path.join(os.curdir, 'projectfiles'))

import matplotlib.pyplot as plt

import neural_network_model as nn
import common.layers        as layers
import common.initializers  as initializers
import common.optimizers    as optimizers


class ANNClassifier(object):
    def __init__(self, input_size, output_size, hidden_size,
                 activation='relu',
                 optimizer=optimizers.SGD(0.01, 0., 0.),
                 loss='CEE',
                 kernel_initializer=initializers.RandomNormal(0., 0.01),
                 bias_initializer=initializers.Zeros(),
                 random_state=None):

        self.input_size   = input_size    # the number of input neurons
        self.output_size  = output_size   # the number of output neurons
        self.hidden_size  = hidden_size   # the number of hidden neurons
        self.activation   = activation    # activation type
        self.optimizer    = optimizer     # optimizer unit
        self.random_state = random_state  # random seed value

        self.model = nn.Sequential(random_state=self.random_state)
        self.model.add(layers.Dense(units=hidden_size, activation=activation, input_shape=input_size,
                                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        self.model.add(layers.Dense(units=output_size, activation='softmax', input_shape=hidden_size,
                                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        self.model.compile(optimizer=self.optimizer, loss=loss)

        self.costs = None

    def fit(self, input_tensor, target_tensor, batch_size=100, epochs=1000):
        self.model.fit(input_tensor=input_tensor, target_tensor=target_tensor, batch_size=batch_size, epochs=epochs)
        self.costs = self.model.costs

        return self

    def predict(self, input_tensor):
        out = self.model.forward(input_tensor=input_tensor)

        return out

    def plot_costs(self, label=None, marker='o', resolution=1, ax=plt):
        if self.costs is None: return
        cost = [self.costs[idx * resolution] for idx in range(len(self.costs) // resolution)]
        ax.plot(range(1, len(cost) + 1), cost, label=label, marker=marker)
