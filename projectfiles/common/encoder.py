import sys, os
sys.path.append(os.path.join(os.curdir, 'common'))

import numpy as np


class encoder(object):
    def __init__(self):
        pass

    def fit_transform(self, data):
        pass

    def transform(self, data):
        pass


class one_hot_encoder(encoder):
    def __init__(self, categorical_features='auto'):
        super(one_hot_encoder, self).__init__()
        self.categorical_features = categorical_features
        self.features = None
        self.drop_first = None

    def fit_transform(self, data, drop_first=True):
        if data.ndim == 0:
            raise ValueError("data needs to be an array")

        if data.ndim == 1:
            data = data.reshape(data.shape[0], 1)

        encoded = [[] for _ in range(data.shape[0])]
        iscategorical = [isinstance(d, (int, np.int64, np.int32, str)) for d in data[0]]
        self.features = []
        self.drop_first = drop_first

        for fidx in range(data.shape[1]):
            if (self.categorical_features == 'auto' and iscategorical[fidx]) or \
                    (self.categorical_features != 'auto' and fidx in self.categorical_features):

                self.features.append([])

                for didx, d in enumerate(data):
                    if d[fidx] not in self.features[-1]:
                        self.features[-1].append(d[fidx])

                for didx, d in enumerate(data):
                    vec = [0 for _ in self.features[-1]]
                    vec[self.features[-1].index(d[fidx])] = 1

                    if drop_first:
                        encoded[didx] += vec[1:]
                    else:
                        encoded[didx] += vec
            else:
                for didx, d in enumerate(data):
                    encoded[didx].append(d[fidx])

        return np.array(encoded)

    def transform(self, data):
        if self.features is None or self.drop_first is None:
            raise TypeError("the encoder is not fit")

        if len(data.shape) == 0:
            raise ValueError("data needs to be an array")

        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)

        encoded = []

        for didx, d in enumerate(data):
            encoded.append([])

            for fidx, f in enumerate(d):
                vec = [0 for _ in self.features[fidx]]

                if f not in self.features[fidx]:
                    raise TypeError(f"for feature({fidx}), '{f}' is not defined")

                vec[self.features[fidx].index(f)] = 1

                if self.drop_first:
                    encoded[-1] += vec[1:]
                else:
                    encoded[-1] += vec

        return np.array(encoded)


if __name__ == '__main__':
    enc = one_hot_encoder()
    data = np.array(['a', 'b', 'c'])
    transformed = enc.fit_transform(data, drop_first=False)
    print(transformed)
    print(transformed[0].shape)