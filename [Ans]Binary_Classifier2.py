##########################################
#   Project Outline: Binary Classifier   #
########################################## 
# 
# 이번 프로젝트에서는 앞서 구현한 아달라인 모델을 개선하고 로지스틱 회귀
# 모델(Logistic Regression model)을 구현하고자 한다. 
# 
# 프로젝트에서는 업계 표준 라이브러리인 Numpy와 Pandas를 이용한다. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from projectfiles.analyzer.graphics import plot_decision_regions

# 아래 코드는 붓꽃 데이터셋을 로드하고 그 중 setosa와 versicolor 클래스만
# 추출하여 학습에 용이한 현태로 변환한 것이다. 그래프로 표현 가능한 형태로
# 만들기 위해 특성은 0번과 2번 특성만 사용하였다.

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

if __name__ == "__main__":
    print(df.head())



#################
#   Problem 1   #
################# 
# 
# 앞선 모델을 구현하면서 머신러닝 모델을 최적화하는 간단한 방법 중 하나인
# 경사 하강법에 대해 알아보았다. 이번에는 경사 하강법을 보완하여 확률적 경사
# 하강법(Stocastic Gradient Descent)를 구현하고자 한다.
# 
# 앞서 구현한 Adaline 모델에서 가중치를 최적화하는 방식을 확률적 경사 하강법으로
# 변경하여라. 확률적 경사 하강법과 경사 하강법의 차이점은 다음과 같다.
# 
#   - 확률적 경사 하강법은 개별 데이터에 대해 가중치를 업데이트한다.
#     경사 하강법에서는 하나의 데이터 샘플에 대해 동시에 가중치를 업데이트한다.
# 
#   - 실시간으로 데이터의 추가가 가능하다. 경사 하강법에서는 한번 모델의 가중치가
#     결정되면 이어서 학습을 진행할 수 없다. 
# 
# 확률적 경사 하강법을 적용한 Adaline모델을 정의하라.
# 
# 클래스의 이름은 AdalineSGD로 한다.
# 
# 앞선 프로젝트에서 구현한 Adaline 클래스와 크게 다르지는 않다. 앞서 구현한 모델을 보완하는
# 차원에서 구현하라.
# 
# update_weight(self, data, label) 메소드의 경우 앞선 프로젝트에서와 차이점이 있다.
# 이번 프로젝트의 경우 data는 개별 데이터, label은 개별 데이터의 정답 레이블을 의미한다.
# 
# fit(self, data, label)에 shuffle옵션을 추가하라. shuffle옵션이 True이면 데이터셋을 반복할 때
# 마다 데이터 셋을 무작위로 섞도록 구현하라. 앞선 프로젝트에서 구현한 shuffle() 함수를 사용하라.
# 모든 메소드는 클래스에 귀속되어야 한다.
# 
# partial_fit(self, data, label)을 추가하라. 이 메소드는 학습이 끝난 모델에 추가적으로 학습을
# 하기 위한 메소드로, 반복 학습을 하지 않는다. 따라서 이때 발생한 손실은 self.costs에 입력하지 않는다.


# 답안:

class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=1000, random_state=1):
        self.eta = eta                    # 학습률
        self.n_iter = n_iter              # 반복 횟수
        self.random_state = random_state  # 랜덤 시드값
        self.costs = None
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


    def shuffle(self, data, label):
        rgen = np.random.RandomState(seed=self.random_state)
        idx = rgen.permutation(len(label))
        return data[idx], label[idx]


    def weight_initializer(self, data):
        rgen = np.random.RandomState(seed=self.random_state)
        return rgen.normal(loc=0.0, scale=0.01, size=data.shape[1]+1)


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

    
    def predict(self, data):
        return np.where(self.activation(self.net_input(data, self.weight)) >= 0, 1, -1)

    
    def plot_costs(self, resolution=1, ax=plt):
        if self.costs is None: return
        cost = [self.costs[idx * resolution] for idx in range(len(self.costs) // resolution)]
        ax.plot(range(1, len(cost) + 1), cost, marker='o')


if __name__ == "__main__":
    # choosing only 'setosa' and 'versicolor'
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)  # labeling 'setosa' as -1 and 'versicolor' as 1
    ynames = {-1: 'setosa', 1: 'versicolor'}

    # distract features of 'setosa' and 'versicolor'
    X = df.iloc[0:100, [0, 2]].values

    # standardization of the samples
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # fitting model
    model_AD = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
    model_AD.fit(X_std, y)

    # plotting results
    if model_AD.weight is not None and model_AD.costs is not None:
        # plotting result
        adaline_result = plt.figure(figsize=(12, 5))
        ax1 = adaline_result.add_subplot(1, 2, 1)
        ax2 = adaline_result.add_subplot(1, 2, 2)

        model_AD.plot_costs(ax=ax1)
        ax1.set_title('1. AdalineSGD: Costs per Epoch')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('costs')

        plot_decision_regions(X_std, y, model_AD, resolution=0.02, names=ynames, ax=ax2)
        ax2.set_title('2. AdalineSGD: Decision Region')
        ax2.set_xlabel('sepal length')
        ax2.set_ylabel('petal length')
        ax2.legend(loc='upper left')

        plt.show()



#################
#   Problem 2   #
################# 
# 
# 이번에는 또 다른 효과적인 이진 분류기 모델인 로지스틱 회귀(Logistic Regression)
# 모델을 구현하고자 한다. 로지스틱 회귀 모델은 확률적 개념을 도입하여 가중치를 갱신하는
# 이진 분류기 모델이다.
# 
# 로지스틱 회귀 모델의 경우 활성화 함수가 아달린과 다르다. 아달린에서는 별다른 활성화함수를
# 사용하지 않았지만, 로지스틱 회귀 모델의 경우 시그모이드 함수를 활성화함수로 사용한다.
# 
# 시그모이드 함수의 정의는 아래 링크를 참조한다:
# Url 로지스틱회귀:   https://ko.wikipedia.org/wiki/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1_%ED%9A%8C%EA%B7%80#%EC%8B%9D
# Url 시그모이드함수: https://ko.wikipedia.org/wiki/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C_%ED%95%A8%EC%88%98 
# 
# 이번 프로젝트에서는 로지스틱 시그모이드 함수를 시그모이드 함수로 정의한다. 시그모이드
# 함수의 정의는 다음과 같다:
# 
#   sigmoid(x) = 1 / (1 + exp(-x))
# 
# 활성화 함수가 시그모이드 함수일 때 비용함수는 교차 엔트로피 에러(Cross Entropy Error, CEE)를
# 사용한다. 교차 엔트로피 에러는 다음과 같이 정의된다.
# 
#   J(w) = -target * log(output) - (1 - target) * log(target - output)
# 
# target은 정답 레이블이고 output은 활성화된 모델 출력으로 다음과 같이 정의된다.
# 
#   output = sigmoid(w*x)
# 
# 교차 엔트로피 에러를 비용함수로 사용하는 이유는 구현의 난이도를 낮추기 위함이다.
# 방금 소개한 로지스틱 회귀 모델의 비용함수를 가중치 벡터 w에 대해 편미분하면 그 결과가
# 앞서 구현한 아달린 모델에서 사용한 비용함수를 편미분한 것과 동일하다. 따라서 가중치를
# 갱신하는데 사용하는 수식 또한 동일하다.
# 
# 앞서 구현한 AdalineSGD 클래스를 이용하여 로지스틱 회귀 모델을 구현하라.
# 클래스의 이름은 LogisticRegression으로 하라.
# 그 외의 나머지 모든 메소드는 그 기능이 동일하다. 


# 답안:

class LogisticRegression(object):
    def __init__(self, eta=0.01, n_iter=1000, random_state=1):
        self.eta = eta                    # learning rate
        self.n_iter = n_iter              # the number of iteration
        self.random_state = random_state  # random seed
        self.costs = None
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

    def shuffle(self, data, label):
        rgen = np.random.RandomState(seed=self.random_state)
        idx = rgen.permutation(len(label))
        return data[idx], label[idx]

    def weight_initializer(self, data):
        rgen = np.random.RandomState(seed=self.random_state)
        return rgen.normal(loc=0.0, scale=0.01, size=data.shape[1] + 1)

    def net_input(self, data, weight):
        return np.dot(data, weight[1:]) + weight[0]

    def activation(self, net_input):
        return 1.0 / (1.0 + np.exp(-np.clip(net_input, -250, 250)))

    def update_weight(self, data, label):
        output = self.activation(self.net_input(data, self.weight))
        error = label - output
        cost = -label * np.log(output) - (1-label) * np.log(1 - output)
        self.weight[1:] += self.eta * error * data
        self.weight[0] += self.eta * error

        return cost

    def predict(self, data):
        return np.where(self.activation(self.net_input(data, self.weight)) >= 0.5, 1, 0)

    def plot_costs(self, resolution=1, ax=plt):
        if self.costs is None: return
        cost = [self.costs[idx * resolution] for idx in range(len(self.costs) // resolution)]
        ax.plot(range(1, len(cost) + 1), cost, marker='o')


if __name__ == "__main__":
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
    model_LR = LogisticRegression(eta=0.01, n_iter=50, random_state=1)
    model_LR.fit(X_std, y)

    if model_LR.weight is not None and model_LR.costs is not None:
        logistic_result = plt.figure(figsize=(12, 5))
        ax1 = logistic_result.add_subplot(1, 2, 1)
        ax2 = logistic_result.add_subplot(1, 2, 2)

        model_LR.plot_costs(ax=ax1, resolution=5)
        ax1.set_title('1. LogisticRegression: Costs per 5 Epoch')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('costs')

        plot_decision_regions(X_std, y, model_LR, resolution=0.02, names=ynames, ax=ax2)
        ax2.set_title('2. LogisticRegression: Decision Region')
        ax2.set_xlabel('sepal length')
        ax2.set_ylabel('petal length')
        ax2.legend(loc='upper left')

        plt.show()
