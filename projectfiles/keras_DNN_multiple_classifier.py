import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import HeNormal, Zeros
from common.encoder import one_hot_encoder
from analyzer.graphics import plot_decision_regions


if __name__ == "__main__":
    # 붓꽃 데이터셋 학습 알고리즘
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:150, 4].values
    ynames = {}

    for idx, lbl in enumerate(np.unique(y)):
        y[y == lbl] = idx
        ynames[idx] = lbl

    y_encoded = one_hot_encoder(categorical_features='auto').fit_transform(y, drop_first=False)

    X = df.iloc[0:150, [0, 2]].values

    # standardization of the samples
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # fitting model
    random_state = 777

    optimizer = SGD(lr=0.01, momentum=0, decay=0)
    bias_initializer = Zeros()

    model = Sequential()
    model.add(Dense(units=3, activation='relu', input_shape=(2, ),
                    kernel_initializer=HeNormal(seed=random_state),
                    bias_initializer=bias_initializer))
    model.add(Dense(units=5, activation='relu',
                    kernel_initializer=HeNormal(seed=random_state),
                    bias_initializer=bias_initializer))
    model.add(Dense(units=3, activation='softmax',
                    kernel_initializer=HeNormal(seed=random_state),
                    bias_initializer=bias_initializer))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    hist = model.fit(X_std, y_encoded, batch_size=100, epochs=500)

    # plotting result
    result = plt.figure(figsize=(12, 5))
    ax1 = result.add_subplot(1, 2, 1)
    ax2 = result.add_subplot(1, 2, 2)

    ax1.plot(hist.history['loss'])
    ax1.set_title('1. DNN multiple classifier test: Costs per Epoch')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('costs')

    plot_decision_regions(X_std, y, model, resolution=0.01, names=ynames, ax=ax2)
    ax2.set_title('2. DNN multiple classifier test: Decision Region')
    ax2.legend(loc='upper left')

    plt.show()
