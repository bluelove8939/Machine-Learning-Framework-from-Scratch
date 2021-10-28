import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from decision_tree import DecisionTreeClassifier
from analyzer.graphics import plot_decision_regions


if __name__ == "__main__":
    # 붓꽃 데이터셋 학습 알고리즘
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:150, 4].values
    ynames = {}

    for idx, lbl in enumerate(np.unique(y)):
        y[np.where(y == lbl)] = idx
        ynames[idx] = lbl

    X = df.iloc[0:150, [0, 1]].values

    # standardization of the samples
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # fitting model
    model = DecisionTreeClassifier(criterion='gini', max_depth=5)
    model.fit(X_std, y, toler=0.0001)

    # plotting result
    result = plt.figure(figsize=(6, 5))
    ax = result.add_subplot()

    plot_decision_regions(X_std, y, model, resolution=0.02, names=ynames, ax=ax)
    ax.set_title('DecisionTreeClassifier test: Decision Region')
    ax.legend(loc='upper left')

    plt.show()