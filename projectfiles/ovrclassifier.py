import numpy as np
import matplotlib.pyplot as plt

import binclassifier


binclassifier_names = {
    "Adaline":            binclassifier.AdalineSGD,
    "LogisticRegression": binclassifier.LogisticRegression
}


class OVRClassifier(object):
    def __init__(self, classifier, eta=0.01, n_iter=1000, random_state=None):
        self.eta = eta                    # learning rate
        self.n_iter = n_iter              # the number of iteration
        self.random_state = random_state  # random seed
        self.submodels = None             # sub models
        self.label_names = None           # label names

        if isinstance(classifier, str):
            if classifier.upper() not in [name.upper() for name in binclassifier_names.keys()]:
                raise TypeError(f"Binary classifier {classifier} not found")

            self.classifier = binclassifier_names[classifier]
        else:
            self.classifier = classifier

    def fit(self, data, label, shuffle=True):
        self.submodels = []
        self.label_names = np.unique(label)

        for idx, name in enumerate(self.label_names):
            processed_label = np.zeros(shape=label.shape)
            processed_label[np.where(label == name)] = 1

            self.submodels.append(self.classifier(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state))
            self.submodels[-1].fit(data, processed_label, shuffle=shuffle)

        return self

    def partial_fit(self, data, label):
        pass

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-np.clip(x, -250, 250)))

    def predict(self, data, one_hot=False):
        if data.ndim > 1:
            prob = np.zeros(shape=(data.shape[0], len(self.submodels)))

            for midx, model in enumerate(self.submodels):
                prob[:, midx] = self.sigmoid(model.output(data))

            if one_hot:
                oh_res = np.zeros(prob.shape)
                for didx in range(data.shape[0]):
                    oh_res[prob[didx, :].argmax()] = 1
                return oh_res

            res = np.array([prob[idx, :].argmax() for idx in range(data.shape[0])])
            return res
        else:
            prob = np.zeros(shape=len(self.submodels))

            for midx, model in enumerate(self.submodels):
                prob[midx] = model.output(data)

            if one_hot:
                oh_res = np.zeros(prob.shape)
                oh_res[prob.argmax()] = 1
                return oh_res

            res = np.where(prob == max(prob))[0][0]
            return res

    def plot_costs(self, resolution=1, ax=plt):
        if self.submodels is None: return
        markers = ('s', 'x', 'o', '^', 'v', '+', 'x')
        for midx, md in enumerate(self.submodels):
            md.plot_costs(label=f"{self.label_names[midx]}", marker=markers[midx], resolution=resolution, ax=ax)
