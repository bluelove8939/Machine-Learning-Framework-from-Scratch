import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ovrclassifier import OVRClassifier
from analyzer.graphics import plot_decision_regions


if __name__ == "__main__":
    # 붓꽃 데이터셋 학습 알고리즘
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # distracting calss labels and encode as integers
    y = df.iloc[0:150, 4].values
    ynames = {}

    for idx, lbl in enumerate(np.unique(y)):
        y[np.where(y == lbl)] = idx
        ynames[idx] = lbl

    # distracting data features
    X = df.iloc[0:150, [0, 2]].values

    # standardization of the samples
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # fitting model
    model = OVRClassifier(classifier='LogisticRegression', eta=0.01, n_iter=1000, random_state=None)
    model.fit(X_std, y, shuffle=True)

    # plotting results
    result = plt.figure(figsize=(12, 5))
    ax1 = result.add_subplot(1, 2, 1)
    ax2 = result.add_subplot(1, 2, 2)

    model.plot_costs(ax=ax1, resolution=50)
    ax1.set_title('1. OVRClassifier test: Costs per Epoch')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('costs')
    ax1.legend(loc='upper right')

    plot_decision_regions(X_std, y, model, resolution=0.02, names=ynames, ax=ax2)
    ax2.set_title('2. OVRClassifier test: Decision Region')
    ax2.legend(loc='upper right')

    plt.show()