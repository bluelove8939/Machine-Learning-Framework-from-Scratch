import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binclassifier import LogisticRegression
from analyzer.graphics import plot_decision_regions


if __name__ == "__main__":
    # 붓꽃 데이터셋 학습 알고리즘
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # choosing only 'setosa' and 'versicolor'
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)  # labeling 'setosa' as 0 and 'versicolor' as 1
    ynames = {0: 'setosa', 1: 'versicolor'}

    # distract features of 'setosa' and 'versicolor'
    X = df.iloc[0:100, [0, 2]].values

    # standardization of the samples
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # fitting model
    model = LogisticRegression(eta=0.01, n_iter=100, random_state=1)
    model.fit(X_std, y)

    # plotting result
    result = plt.figure(figsize=(12, 5))
    ax1 = result.add_subplot(1, 2, 1)
    ax2 = result.add_subplot(1, 2, 2)

    model.plot_costs(ax=ax1, resolution=10)
    ax1.set_title('1. LogisticRegression test: Costs per Epoch')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('costs')

    plot_decision_regions(X_std, y, model, resolution=0.02, names=ynames, ax=ax2)
    ax2.set_title('2. LogisticRegression test: Decision Region')
    ax2.set_xlabel('sepal length')
    ax2.set_ylabel('petal length')
    ax2.legend(loc='upper left')

    plt.show()