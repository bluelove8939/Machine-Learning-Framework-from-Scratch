import numpy as np


def gini(s):
    ret = 1
    for element in np.unique(s):
        ret -= (np.count_nonzero(s == element) / s.shape[0]) ** 2
    return ret

def cross_entropy(s):
    ret = 0
    for element in np.unique(s):
        p = np.count_nonzero(s == element) / s.shape[0]
        if p > 0:
            ret -= p * np.log2(p)
    return ret


criterion_names = {'gini': gini, 'entropy': cross_entropy}


class DecisionTreeClassifier(object):
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion    # criterion type ex) gini, cross_entropy ...
        self.max_depth = max_depth    # maximum depth of the tree

        if isinstance(self.criterion, str):
            self.criterion = self.criterion.lower()

            if self.criterion not in criterion_names.keys():
                raise TypeError(f"criterion {self.criterion} not found")

            self.criterion = criterion_names[self.criterion]

        self.dominant = None  # dominant class label
        self.impurity = None  # impurity of the node
        self.fidx = None   # index of the feature
        self.value = None
        self.left = None   # linked node (feature['fidx'] is less than 'value')
        self.right = None  # linked node (feature['fidx'] is greater or equals to 'value')

    def fit(self, data, label, toler=0.0001, didx=None):
        if didx is None:
            didx = np.array([True] * data.shape[0])

        # calculate impurity of the node and find out the domianat class label
        label_name, label_counts = np.unique(label[didx], return_counts=True)
        self.impurity = self.criterion(label[didx])
        self.dominant = label_name[np.where(label_counts == max(label_counts))[0][0]]
        ct = np.count_nonzero(didx)  # the number of data samples in this node

        if ct == 0 or np.unique(label[didx]).shape[0] <= 1:
            return self

        if self.max_depth is not None and self.max_depth == 0:
            return self

        # feature that minimizes the impurity
        impurity_min = 10
        lidx, ridx = None, None

        for pivot, name1 in enumerate(label_name):
            for name2 in label_name[pivot+1:]:
                for fidx in range(data.shape[1]):
                    d1 = data[didx & (label == name1), fidx]
                    d2 = data[didx & (label == name2), fidx]

                    d1_q1, d1_q3 = np.percentile(d1, 25), np.percentile(d1, 75)
                    d2_q1, d2_q3 = np.percentile(d2, 25), np.percentile(d2, 75)

                    candidate = (max(d1_q1, d2_q1) + min(d1_q3, d2_q3)) / 2

                    idx1 = didx & (data[:, fidx] <  candidate)
                    idx2 = didx & (data[:, fidx] >= candidate)

                    g1 = self.criterion(label[idx1])
                    g2 = self.criterion(label[idx2])

                    ct1 = np.count_nonzero(idx1)
                    ct2 = np.count_nonzero(idx2)

                    g = (ct1 / ct) * g1 + (ct2 / ct) * g2

                    if g < impurity_min:
                        impurity_min = g
                        self.fidx = fidx
                        self.value = candidate
                        lidx, ridx = idx1, idx2

        if self.impurity - impurity_min < toler:
            return self

        nxt_max_depth = self.max_depth

        if self.max_depth is not None:
            nxt_max_depth = nxt_max_depth - 1

        self.left  = DecisionTreeClassifier(criterion=self.criterion, max_depth=nxt_max_depth)
        self.right = DecisionTreeClassifier(criterion=self.criterion, max_depth=nxt_max_depth)

        self.left.fit(data, label, toler=toler, didx=lidx)
        self.right.fit(data, label, toler=toler, didx=ridx)

        return self

    def predict(self, data):
        if data.ndim == 1:
            if self.right is None or self.left is None:
                return self.dominant

            return self.left.predict(data) if data[self.fidx] < self.value else self.right.predict(data)

        if self.right is None or self.left is None:
            return np.array([self.dominant] * data.shape[0])

        d1_idx = data[:, self.fidx] <  self.value
        d2_idx = data[:, self.fidx] >= self.value

        l1 = self.left.predict(data[d1_idx])
        l2 = self.right.predict(data[d2_idx])

        label = np.zeros(shape=data.shape[0])
        label[d1_idx] = l1
        label[d2_idx] = l2

        return label


class RandomForest(object):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, max_features='auto', random_state=None):
        self.n_estimators = n_estimators  # the number of submodels
        self.criterion = criterion        # criterion type ex) gini, cross_entropy ...
        self.max_depth = max_depth        # maximum depth of the tree
        self.max_features = max_features  # the number of maximum features to select
        self.random_state = random_state  # random seed
        self.rgen = np.random.RandomState(self.random_state)  # random number generator

        if isinstance(self.criterion, str):
            self.criterion = self.criterion.lower()

            if self.criterion not in criterion_names.keys():
                raise TypeError(f"criterion {self.criterion} not found")

            self.criterion = criterion_names[self.criterion]

        self.submodels = [DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)
                          for _ in range(self.n_estimators)]
        self.feature_idx = [None for _ in range(self.n_estimators)]
        self.label_names = None

    def fit(self, data, label, toler=0.001):
        self.label_names = np.unique(label)

        n_features = int(np.sqrt(data.shape[1]))

        if self.max_features == 'log2':
            n_features = int(np.log2(data.shape[1]))
        elif self.max_features is None:
            n_features = int(data.shape[1])

        for midx, model in enumerate(self.submodels):
            didx = self.rgen.choice(range(0, data.shape[0], 1), size=data.shape[0])
            fidx = self.rgen.choice(range(0, data.shape[1], 1), replace=False, size=n_features)
            boot_data, boot_label = data[didx][:, fidx], label[didx]
            self.feature_idx[midx] = fidx
            model.fit(boot_data, boot_label, toler=toler)

        return self

    def predict(self, data):
        if len(data.shape) == 1:
            predicted_labels = [model.predict(data[fidx]) for fidx, model in zip(self.feature_idx, self.submodels)]
            counts, labels = np.unique(predicted_labels)
            return labels[counts.argmax()]

        voting = np.zeros(shape=(data.shape[0], self.label_names.shape[0]))

        for fidx, model in zip(self.feature_idx, self.submodels):
            data_distracted = data[:, fidx]
            label_predicted = model.predict(data_distracted)
            for lidx, lbl in enumerate(self.label_names):
                voting[np.where(label_predicted == lbl), lidx] += 1

        return np.array([self.label_names[vec.argmax()] for vec in voting])


if __name__ == "__main__":
    pass
