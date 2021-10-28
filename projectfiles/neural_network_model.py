import sys, os
sys.path.append(os.path.join(os.curdir, 'projectfiles'))

import numpy as np
import matplotlib.pyplot as plt

import common.optimizers   as optimizers
import common.layers       as layers


class Sequential(object):
    def __init__(self, random_state=None):
        self.seq    = []    # sequence of layers
        self.opt    = None  # optimizer module
        self.loss   = None  # loss layer
        self.costs  = None  # costs
        self.out    = None  # output result of forward propagation
        self.rgen   = np.random.RandomState(seed=random_state)  # random number generator

    def add(self, ly):
        if ly.input_shape is None:
            if len(self.seq) == 0:
                raise TypeError("Input shape of first layer is not defined")
            ly.input_shape = self.seq[-1].units
        else:
            if len(self.seq) != 0 and self.seq[-1].units != ly.input_shape:
                raise TypeError(f"Input shape {ly.input_shape} is not the same with previous layer")

        self.seq.append(ly)

    def compile(self, optimizer=optimizers.SGD(), loss='mse'):
        if not isinstance(loss, str):
            raise TypeError("'loss' needs to be a string")

        if loss not in layers.loss_units.keys():
            raise TypeError(f"loss name {loss} not found")

        # define optimizer and loss layer
        self.opt  = optimizer
        self.loss = layers.loss_units[loss.lower()]()

        # compile all the layers
        for ly in self.seq:
            ly.compile()

    def forward(self, input_tensor):
        self.out = input_tensor
        for ly in self.seq: self.out = ly.forward(self.out)

        return self.out

    def backward(self, target_tensor):
        loss = self.loss.forward(self.out, target_tensor)
        dout = self.loss.backward(dout=1)
        for ly in reversed(self.seq): dout = ly.backward(dout)

        return loss

    def optimize(self):
        params, grads = [], []
        cnt = 0

        for ly in self.seq:
            if isinstance(ly.params, list) and isinstance(ly.grads, list):
                params += ly.params
                grads  += ly.grads
                cnt += 1
            else:
                params.append(ly.params)
                grads.append(ly.grads)

        self.opt(params, grads)

        return self

    def fit(self, input_tensor, target_tensor, batch_size=100, epochs=1000, **kwargs):
        kwargs_list = {'verbose': 0}

        for arg_name in kwargs.keys():
            if arg_name not in kwargs_list.keys():
                raise TypeError(f"Unknown keyword argument {arg_name}")
            kwargs_list[arg_name] = kwargs[arg_name]


        self.costs = []
        train_size = input_tensor.shape[0]  # size of the test dataset

        for e in range(epochs):
            if kwargs_list['verbose'] == 1:
                print(f"epoch: {e+1} started...", end="")
            batch_mask   = self.rgen.choice(train_size, batch_size)
            input_batch  = input_tensor[batch_mask]
            target_batch = target_tensor[batch_mask]

            if kwargs_list['verbose'] == 1:
                print(f"\repoch: {e+1} forward propagation...", end="")
            self.forward(input_batch)

            if kwargs_list['verbose'] == 1:
                print(f"\repoch: {e+1} backward propagation...", end="")
            cost = self.backward(target_batch)
            self.costs.append(cost.mean())

            if kwargs_list['verbose'] == 1:
                print(f"\repoch: {e+1} optimization...", end="")
            self.optimize()

            if kwargs_list['verbose'] == 1:
                print(f"\repoch: {e+1} done => average loss: {cost}")

        return self

    def predict(self, input_tensor, one_hot=False):
        out = input_tensor

        for ly in self.seq:
            out = ly.forward(out)

        if not one_hot:
            return np.array([out[idx, :].argmax() for idx in range(input_tensor.shape[0])])
        return out

    def plot_costs(self, label=None, marker='o', resolution=1, ax=plt):
        if self.costs is None: return
        cost = [self.costs[idx * resolution] for idx in range(len(self.costs) // resolution)]
        ax.plot(range(1, len(cost) + 1), cost, label=label, marker=marker)
