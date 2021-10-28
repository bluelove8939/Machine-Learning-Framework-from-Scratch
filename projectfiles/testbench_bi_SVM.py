import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binclassifier import SVM
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
    model = SVM(random_state=1)
    model.fit(X_std, y, C=0.6, toler=0.001, max_iter=40)

    # plotting result
    result = plt.figure(figsize=(6, 5))
    ax = result.add_subplot()

    plot_decision_regions(X_std, y, model, resolution=0.02, names=ynames, ax=ax)
    ax.set_title('SVM test: Decision Region')
    ax.set_xlabel('sepal length')
    ax.set_ylabel('petal length')
    ax.legend(loc='upper left')

    plt.show()