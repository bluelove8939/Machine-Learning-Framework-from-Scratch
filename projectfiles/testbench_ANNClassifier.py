import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from annclassifier import ANNClassifier
from common.optimizers import SGD
from common.encoder import one_hot_encoder
from common.initializers import RandomNormal, Zeros
from analyzer.graphics import plot_decision_regions


if __name__ == "__main__":
    # 붓꽃 데이터셋 학습 알고리즘
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    ynames = {}

    for idx, lbl in enumerate(np.unique(y)):
        y[y == lbl] = idx
        ynames[idx] = lbl

    y_encoded = one_hot_encoder(categorical_features='auto').fit_transform(y, drop_first=False)

    X = df.iloc[0:100, [0, 2]].values

    # standardization of the samples
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # fitting model
    optimizer = SGD(lr=0.01, momentum=0, decay=0)
    kernel_initializer = RandomNormal(mean=0., stddev=0.01, random_state=1)
    bias_initializer   = Zeros()

    input_size = X_std.shape[1]
    output_size = y_encoded.shape[1]
    hidden_size = 5

    model = ANNClassifier(input_size=input_size,
                          output_size=output_size,
                          hidden_size=hidden_size,
                          activation='relu', loss='cee',
                          optimizer=optimizer,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          random_state=1)
    model.fit(X_std, y_encoded, batch_size=100, epochs=500)

    # plotting result
    result = plt.figure(figsize=(12, 5))
    ax1 = result.add_subplot(1, 2, 1)
    ax2 = result.add_subplot(1, 2, 2)

    model.plot_costs(ax=ax1, resolution=25)
    ax1.set_title('1. ANNClassifier test: Costs per Epoch')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('costs')

    plot_decision_regions(X_std, y, model, resolution=0.02, names=ynames, ax=ax2)
    ax2.set_title('2. ANNClassifier test: Decision Region')
    ax2.legend(loc='upper left')

    plt.show()
