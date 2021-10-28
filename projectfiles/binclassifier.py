import numpy as np
import matplotlib.pyplot as plt


class BinClassifier(object):
    def __init__(self, random_state):
        self.random_state = random_state
        self.costs = None

    def shuffle(self, data, label):
        rgen = np.random.RandomState(seed=self.random_state)
        idx = rgen.permutation(len(label))
        return data[idx], label[idx]

    def weight_initializer(self, data):
        rgen = np.random.RandomState(seed=self.random_state)
        return rgen.normal(loc=0.0, scale=0.01, size=data.shape[1] + 1)

    def plot_costs(self, label=None, marker='o', resolution=1, ax=plt):
        if self.costs is None: return
        cost = [self.costs[idx * resolution] for idx in range(len(self.costs) // resolution)]
        ax.plot(range(1, len(cost) + 1), cost, label=label, marker=marker)


class AdalineGD(BinClassifier):
    def __init__(self, eta=0.01, n_iter=1000, random_state=None):
        super(AdalineGD, self).__init__(random_state=random_state)

        self.eta = eta  # learning rate
        self.n_iter = n_iter  # the number of iteration
        self.weight = None

    def fit(self, data, label):
        self.weight = self.weight_initializer(data)
        self.costs = list()

        for _ in range(self.n_iter):
            self.costs.append(self.update_weight(data, label))

        return self

    def net_input(self, data, weight):
        return np.dot(data, weight[1:]) + weight[0]

    def activation(self, net_input):
        return net_input

    def update_weight(self, data, label):
        output = self.activation(self.net_input(data, self.weight))
        errors = label - output
        cost = (errors ** 2).sum() / 2.0
        self.weight[1:] += self.eta * data.T.dot(errors)
        self.weight[0] += self.eta * errors.sum()

        return cost

    def output(self, data):
        return self.net_input(data, self.weight)

    def predict(self, data):
        return np.where(self.activation(self.net_input(data, self.weight)) >= 0.5, 1, 0)


class AdalineSGD(BinClassifier):
    def __init__(self, eta=0.01, n_iter=1000, random_state=None):
        super(AdalineSGD, self).__init__(random_state=random_state)

        self.eta = eta  # learning rate
        self.n_iter = n_iter  # the number of iteration
        self.weight = None

    def fit(self, data, label, shuffle=True):
        self.weight = self.weight_initializer(data)
        self.costs = []

        for _ in range(self.n_iter):
            if shuffle:
                data, label = self.shuffle(data, label)

            cost = []
            for d, l in zip(data, label):
                cost.append(self.update_weight(d, l))
            self.costs.append(sum(cost) / len(label))

        return self

    def partial_fit(self, data, label):
        if self.weight is None:
            raise TypeError("Classifier is not fit")

        if label.ravel().shape[0] > 1:
            for d, l in zip(data, label):
                self.update_weight(d, l)
        else:
            self.update_weight(data, label)

        return self

    def net_input(self, data, weight):
        return np.dot(data, weight[1:]) + weight[0]

    def activation(self, net_input):
        return net_input

    def update_weight(self, data, label):
        output = self.activation(self.net_input(data, self.weight))
        error = label - output
        cost = (error ** 2) / 2.0
        self.weight[1:] += self.eta * error * data
        self.weight[0] += self.eta * error

        return cost

    def output(self, data):
        return self.net_input(data, self.weight)

    def predict(self, data):
        return np.where(self.activation(self.net_input(data, self.weight)) >= 0, 1, -1)


class LogisticRegression(BinClassifier):
    def __init__(self, eta=0.01, n_iter=1000, random_state=None):
        super(LogisticRegression, self).__init__(random_state=random_state)

        self.eta = eta  # learning rate
        self.n_iter = n_iter  # the number of iteration
        self.weight = None

    def fit(self, data, label, shuffle=True):
        self.weight = self.weight_initializer(data)
        self.costs = []

        for _ in range(self.n_iter):
            if shuffle:
                data, label = self.shuffle(data, label)

            cost = []
            for d, l in zip(data, label):
                cost.append(self.update_weight(d, l))
            self.costs.append(sum(cost) / len(label))

        return self

    def partial_fit(self, data, label):
        if self.weight is None:
            raise TypeError("Classifier is not fit")

        if label.ravel().shape[0] > 1:
            for d, l in zip(data, label):
                self.update_weight(d, l)
        else:
            self.update_weight(data, label)

        return self

    def net_input(self, data, weight):
        return np.dot(data, weight[1:]) + weight[0]

    def activation(self, net_input):
        return 1. / (1. + np.exp(-np.clip(net_input, -250, 250)))

    def update_weight(self, data, label):
        output = self.activation(self.net_input(data, self.weight))
        error = label - output
        cost = -label * np.log(output) - (1-label) * np.log(1 - output)
        self.weight[1:] += self.eta * error * data
        self.weight[0] += self.eta * error

        return cost

    def output(self, data):
        return self.net_input(data, self.weight)

    def predict(self, data):
        return np.where(self.activation(self.net_input(data, self.weight)) >= 0.5, 1, 0)


class SVM(BinClassifier):
    def __init__(self, n_iter=1000, random_state=None):
        super(SVM, self).__init__(random_state=random_state)

        self.n_iter = n_iter     # the number of iteration (regulation)
        self.label_names = None  # label names (0: positive, 1: negative)
        self.weight = None       # weight vector

    def fit(self, data, label, C, toler, max_iter):
        # Simplified SMO(Sequential Minimal Optimazation) algorithm
        self.label_names = np.unique(label)

        if len(self.label_names) > 2:
            raise TypeError("SVM cannnot classify more than 2 class labels")

        alpha = np.zeros(shape=data.shape[0])  # lagrange multipliers
        thres = 0                              # threshold value
        lbl   = np.zeros(shape=label.shape)    # integer label (positive: 1, negative: -1)
        rgen  = np.random.RandomState(seed=self.random_state)   # random number generator

        lbl[np.where(label == self.label_names[0])] =  1
        lbl[np.where(label == self.label_names[1])] = -1

        cnt_iter = 0
        cnt_overall_iter = 0

        while cnt_iter < max_iter and cnt_overall_iter < self.n_iter:
            cnt_changes = 0

            for i in range(alpha.shape[0]):
                # optimize by alpha[i]
                tf_i = np.sum(lbl * alpha * np.dot(data, data[i, :])) + thres
                E_i = tf_i - lbl[i]

                if (lbl[i] * E_i < -toler and alpha[i] < C) or (lbl[i] * E_i > toler and alpha[i] > 0):
                    # choose random data[j]
                    j = i
                    while j == i or np.array_equal(data[i, :], data[j, :]):
                        j = int(rgen.uniform(0, data.shape[0]))

                    # optimize alpha[j] with alpha[i]
                    tf_j = np.sum(lbl * alpha * np.dot(data, data[j, :])) + thres
                    E_j = tf_j - lbl[j]
                    eta = 2 * np.dot(data[j, :], data[i, :]) - np.dot(data[i, :], data[i, :])    \
                                                             - np.dot(data[j, :], data[j, :])
                    alpha_i_old, alpha_j_old = alpha[i].copy(), alpha[j].copy()

                    if eta == 0:
                        continue

                    # define minima and maxima
                    if lbl[i] != lbl[j]:
                        minima = max(0, alpha[j] - alpha[i])
                        maxima = min(C, C + alpha[j] - alpha[i])
                    else:
                        minima = max(0, alpha[j] + alpha[i] - C)
                        maxima = min(C, alpha[j] + alpha[i])

                    # update alpha[j]
                    alpha[j] -= lbl[j] * (E_i - E_j) / eta
                    if alpha[j] > maxima: alpha[j] = maxima
                    if alpha[j] < minima: alpha[j] = minima

                    # update alpha[i] with the same amount to the opposite direction
                    alpha[i] += lbl[j] * lbl[i] * (alpha_j_old - alpha[j])

                    # if there's no sufficient amount of difference between alpha[j] and alpha_j_old
                    # do not update the threshold and break the sequence
                    if abs(alpha[j] - alpha_j_old) < 0.00001:
                        continue

                    # update threshold
                    if 0 <  alpha[i] < C:
                        thres -= E_i + lbl[i] * (alpha[i] - alpha_i_old) + lbl[j] * (alpha[j] - alpha_j_old)
                    elif 0 < alpha[j] < C:
                        thres -= E_j + lbl[i] * (alpha[i] - alpha_i_old) + lbl[j] * (alpha[j] - alpha_j_old)
                    else:
                        thres -= (E_i + E_j) / 2 + lbl[i] * (alpha[i] - alpha_i_old) + lbl[j] * (alpha[j] - alpha_j_old)

                    cnt_changes += 1

            if cnt_changes == 0:
                cnt_iter += 1
            else:
                cnt_iter = 0

            cnt_overall_iter += 1

        # calculate weight vector (threshold value is saved as weight[0])
        self.weight = np.zeros(shape=data.shape[1]+1)
        self.weight[1:] = np.dot(alpha * lbl, data)
        self.weight[0]  = thres

        return self

    def net_input(self, data, weight):
        return np.dot(data, weight[1:]) + weight[0]

    def output(self, data):
        return self.net_input(data, self.weight)

    def predict(self, data):
        return np.where(self.net_input(data, self.weight) >= 0, self.label_names[0], self.label_names[1])
