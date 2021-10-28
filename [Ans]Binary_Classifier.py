##########################################
#   Project Outline: Binary Classifier   #
########################################## 
# 
# 이번 프로젝트에서는 머신러닝에서 사용되는 이진 분류기(binary classifier)를
# 구현하고자 한다. 이진 분류기는 지도 학습에 사용될 수 있으며, 테스트 데이터셋을
# 바탕으로 모델을 최적화한다.
# 
# 이진 분류기를 이용하면 데이터를 레이블 단위로 분류할 수 있다. 이번 프로젝트에서는
# 이진 분류기의 가장 기본적인 형태인 아달린(Adaline)을 구현하고자 한다. 
# 
# 프로젝트에서는 업계 표준 라이브러리인 Numpy와 Pandas를 이용한다. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from projectfiles.analyzer.graphics import plot_decision_regions



#################
#   Problem 1   #
################# 
# 
# 이진 분류기의 기본적인 동작원리는 일차원 데이터를 실수 데이터로 차원 감소시키는 것이다.
# 이를 위해서 데이터가 다음과 같이 주어질 때, 가중치 벡터 weight를 다음과 같이 정의한다. 

data = np.array([
    [0.7, 0.8, 0.1, 0.2, 0.5, 0.4, 0.5, 0.4, 0.3, 0.2],
    [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.5, 0.1, 0.2, 0.1, 0.5, 0.4, 0.3, 0.3, 0.1, 0.2]
])

rgen = np.random.RandomState(seed=1)   # 난수 생성기

def weight_initializer(data):
    return rgen.normal(loc=0.0, scale=0.01, size=data.shape[1]+1)   # 가중치 벡터 초기화

weight = weight_initializer(data)

# 위에서 정의된 weight는 가중치 벡터로 weight_initializer()에 의해 다음과 같은
# 스펙으로 초기화 된다:
#
#   평균이 0이고 표본분산이 0.01이며 길이가 (개별 데이터의 길이)+1인 정규화된 임의의 벡터
#
# rgen은 난수 생성기로 특정 시드(seed)값에 대한 난수를 생성해주는 난수 생성기이다.
# 만약 특정 시드값을 줄 필요가 없으면 굳이 난수 생성기를 사용하지 않고 다음과 같이
# 가중치 벡터를 초기화할 수 있다:
#
#   weight = np.random.normal(loc=0.0, scale=0.01, size=data.shape[0]+1)
#
# 하지만 프로젝트에서는 일관된 결과를 도출하기 위해 특정 시드값(위에서는 1)을 사용하여
# 난수를 생성하는 것으로 한다.
#
# 예제 원본 데이터는 다음과 같다.

data = np.array([
    [0.7, 0.8, 0.1, 0.2, 0.5, 0.4, 0.5, 0.4, 0.3, 0.2],
    [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.5, 0.1, 0.2, 0.1, 0.5, 0.4, 0.3, 0.3, 0.1, 0.2]
])

# 이때 출력 output은 다음과 같이 정의된다.
#
#   output = (data * weight[1:]) + weight[0]
#
# '*'은 두 벡터간의 내적을 계산하는 연산자이다. 따라서 output은 하나의 실수가 되며,
# 일차원 원본 데이터에 대해 성공적으로 차원감소가 이루어졌음을 알 수 있다.
#
# 데이터가 일차원 데이터라는 것에 유의한다. data는 이차원이지만 이는 여러개의 데이터가
# 묶인 데이터 샘플이기 때문이다. 지금 구현하고자 하는 함수는 하나의 개별 데이터에도
# 맞는 결과를 리턴해야 하지만, 데이터 샘플에도 맞는 결과를 리턴해야 한다. 다행이도
# numpy에서 제공하는 거의 대부분의 연산에는 브로드캐스팅(broadcasting)기능이 지원된다.
# 이를 잘 활용하여 데이터 샘플에도 맞는 결과를 리턴하는 함수를 작성한다.
#
# data와 weight가 주어졌을 때, output을 계산하는 함수를 작성하여라.
# 함수의 이름은 net_input(data, weight)으로 하라.
# data는 원본 데이터 샘플을 의미한다.
# weight는 가중치 벡터를 의미한다.
# weight의 길이는 항상 data의 길이보다 1 길다고 가정한다.
# numpy 에서 내적을 구하는 함수를 활용할 것
#
# 예제에 대한 출력: 0.0035 0.0111 0.0082
  

# 답안:

def net_input(data, weight):
    return np.dot(data, weight[1:]) + weight[0]


if __name__ == "__main__":
    print(f"\nCheck net_input(): {' '.join(['{:0.4f}'.format(num) for num in net_input(data, weight)])}")



#################
#   Problem 2   #
################# 
# 
# 일반적으로 머신 러닝 모델에서 출력(output)은 입력을 활성화시킨 것을 의미한다.
# 하지만 지금 구현하고자 하는 모델의 경우 활성화 과정이 굳이 필요하지는 않다.
# 
# 그래도 앞으로의 구현을 쉽게 하기 위해 활성화 단계를 미리 구현하고자 한다.
# 이번 모델에서 활성화 단계는 항등함수로 정의된다.
# 
# 활성화 단계를 위한 함수를 작성하라.
# 함수의 이름은 activation(data)로 하라.
# data는 원본 데이터 샘플을 의미한다.


# 답안:

def activation(net_input):
    return net_input



#################
#   Problem 3   #
################# 
# 
# 이진 분류기는 어떤 데이터가 해당 분류에 속하는지 혹은 속하지 않는지를 판별하는
# 분류기이다. 원본 데이터의 차원감소가 성공적으로 이루어졌다면, 해당 데이터를 가지고
# 실제 데이터가 어느 분류에 속하는지를 결정지어야 한다.
# 
# 이번 프로젝트에서는 이를 위한 아주 간단한 해결책을 제시한다:
# 
#   만약 출력(output = activation(net_input()))이 0보다 크다면 1번 레이블이라고 결정하고,
#   아니면 -1번 레이블로 결정한다.
# 
# 이와 같은 간단한 레이블 결정 방법은 선형 이진 분류기 모델에서 많이 사용되는 기법이다.
# 
# 함수의 이름은 predict(data, weight) 으로 한다.
# data는 원본 데이터 샘플을 의미한다.
# weight는 가중치 벡터를 의미한다.
# (예를 들어 출력이 [0.1, 0.1, -0.1, -0.1]이면 함수의 리턴값은 [1, 1, -1, -1]이다)
# np.where()함수를 이용하라.
# 
# 예제에 대한 출력: 1.0000 1.0000 1.0000


# 답안:

def predict(data, weight):
    return np.where(activation(net_input(data, weight)) >= 0, 1, -1)
    

if __name__ == "__main__":
    print(f"\nCheck predict(data, weight): " + \
          f"{' '.join(['{:0.4f}'.format(num) for num in predict(data, weight)])}")



#################
#   Problem 4   #
################# 
# 
# 일반적으로 데이터는 개별로 주어지지 않고 샘플 단위로 주어지는 경우가 대부분이다.
# 따라서 대부분의 경우 입력 데이터는 아래와 같이 여러개의 원본 벡터가 일렬로 늘어선
# 형태(행렬)를 띠게 된다.

# data = np.array([
#     [0.7, 0.8, 0.1, 0.2, 0.5, 0.4, 0.5, 0.4, 0.3, 0.2],
#     [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
#     [0.5, 0.1, 0.2, 0.1, 0.5, 0.4, 0.3, 0.3, 0.1, 0.2]
# ])

# 결국 앞서 구현한 net_input() 함수와 predict() 함수의 결과값은 1차원 벡터가 된다.
# 앞으로 이 점에 유의하여 구현을 진행해야 한다. (혹시 앞에서 구현한 함수들이 데이터 샘플에
# 대해 유의미한 결과를 리턴하지 않는다면 지금 이를 수정한다)
# 
# 보통의 머신러닝 혹은 딥러닝 모델들은 하나의 샘플을 여러차례 학습한다. 어떠한 샘플을
# 한 차례 학습하는 것을 에폭(epoch)이라 한다.
# 
# 하나의 샘플을 학습할 때마다 샘플 데이터를 무작위로 섞고자 한다. 이러한 작업은 가중치를
# 갱신할 때 순서에 대한 연관성을 배제함으로서 보다 신뢰도 있는 결과가 도출될 수 있게 한다.
# 
# 데이터 샘플을 섞을 때는 개별 데이터의 정답 레이블(각각의 데이터가 실제로 속해있는 분류)도
# 같은 순서로 섞여야 한다. 그래야 해당 데이터의 레이블을 추측한 뒤 정답을 확인할 수 있기
# 때문이다. 
# 
# 샘플 데이터와 레이블이 다음과 같이 주어질 때 샘플을 섞는 함수를 구현하라.
# 
# 함수의 이름은 shuffle(data, label)로 하라.
# data는 원본 데이터 샘플을 의미한다.
# label은 데이터 샘플의 정답 레이블을 의미한다.
# label[i]는 data[i]의 정답 레이블로 -1 아니면 1이다.
# 
# 이 함수는 원본 샘플 데이터를 훼손해서는 안된다.
# 대신 새롭게 섞인 샘플 데이터를 리턴해야 한다. 
# 
# np.random.permutation() 함수를 사용하여 구현할 것.
# (단, 위에서 정의한 난수 생성기 rgen을 이용하여 호출해야 한다) 


data = np.array([
    [0.7, 0.8, 0.1, 0.2, 0.5, 0.4, 0.5, 0.4, 0.3, 0.2],
    [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.5, 0.1, 0.2, 0.1, 0.5, 0.4, 0.3, 0.3, 0.1, 0.2]
])

label = np.array([-1, 1, -1])


# 답안:

def shuffle(data, label):
    shuffled_idx = rgen.permutation(len(label))
    return data[shuffled_idx], label[shuffled_idx]


if __name__ == "__main__":
    print("\nCheck shuffle(data, label): ")

    s_data, s_label = shuffle(data, label)
    print("\nraw data \n" + '\n'.join([f"data: {d}  label: {l}" for d, l in zip(data, label)]))
    print("\nshuffled data: \n" + '\n'.join([f"data: {sd}  label: {sl}" for sd, sl in zip(s_data, s_label)]))



#################
#   Problem 5   #
################# 
# 
# 이진 분류기는 기본적으로 다음과 같은 과정을 통해 동작한다:
# 
#   1. 주어진 데이터셋의 레이블을 예측한다.
#   2. 예측된 결과값과 정답 레이블 사이의 오차를 계산한다.
#   3. 구한 오차를 바탕으로 비용함수를 계산한 뒤, 비용함수를 최소화하는 방향으로
#      가중치를 갱신한다.
#   4. 이를 주어진 횟수만큼 반복한다.
# 
# 앞서 구한 net_input() 함수와 predict() 함수를 조합하면 주어진 데이터의
# 레이블을 예측할 수 있다. 이제는 정답 레이블과 비교하여 오차를 계산한 후,
# 이를 바탕으로 가중치 벡터를 갱신하는 함수를 만들고자 한다.
# 
# 머신러닝 모델을 최적화하는데 사용되는 기법에는 여러가지가 있다. 이번에는 경사 하강법
# (Gradient Descent)을 사용하고자 한다. 이에 앞서 먼저 비용함수를 결정한다. 비용함수는
# 정답과 예측값 사이의 오차를 바탕으로 현재 가중치가 정답에 얼마나 근접해있는지를 보여준다.
# 
# 이번에 사용할 비용함수는 제곱 오차합(Sum of Squared Error, SSE)이다. 제곱 오차합은
# 데이터 샘플의 비용을 다음과 같이 계산한다.
#   
#   1. 실제 레이블과 예측 레이블 사이의 차이를 계산한다.
#   2. 각각의 데이터에 대한 오차를 제곱한 후 이를 모두 더한다.
#   3. 더한 값을 2로 나눈다.
# 
# 가중치 갱신에는 경사 하강법(Gradient Descent, GD)를 이용하고자 한다. 경사 하강법은
# 비용함수가 최소가 되는 가중치 벡터를 계산하는 방법 중 하나로, 기울기가 감소하는 방향으로
# 가중치를 갱신함으로써 최소값(실질적으로는 극소값)을 찾는다. 이에 대한 자세한 내용은 다음
# 링크를 참조한다.
# Url: https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95
# Url: https://angeloyeo.github.io/2020/08/16/gradient_descent.html
# 
# 아래 링크는 가중치 갱신을 쉽게 설명한 블로그 글이다. 이를 참고하여 구현한다.
# Url: https://brunch.co.kr/@hvnpoet/64   
# 
# 비용함수와 그 미분값을 통해 경사 하강법 알고리즘을 수행하는 함수를 작성하라.
#
# 함수의 이름은 update_weight(data, target)으로 하라.
# data는 원본 데이터 샘플이다. (n_samples, attributes)
# label은 각 데이터 샘플의 정답 레이블이다. 레이블은 1 혹은 -1이다. (n_labels)
# 각 데이터 샘플의 비용을 리턴하라. (n_costs)
# 데이터와 초기 가중치는 아래 정의된 것을 이용하라.
# 원본 데이터의 정답 레이블은 아래 정의된 것을 이용하라. 
# 학습률(learning rate)은 아래 주어진 값을 이용하라.

learning_rate = 0.01

label = np.array([-1, 1, -1])

data = np.array([
    [0.7, 0.8, 0.1, 0.2, 0.5, 0.4, 0.5, 0.4, 0.3, 0.2],
    [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    [0.5, 0.1, 0.2, 0.1, 0.5, 0.4, 0.3, 0.3, 0.1, 0.2]
])

weight = weight_initializer(data)


# 답안:

def update_weight(data, label):
    # 1. 출력을 계산
    output = activation(net_input(data, weight))

    # 2. 출력을 바탕으로 오차를 계산
    errors = label - output

    # 3. 비용을 계산하고 가중치를 갱신
    cost = (errors ** 2).sum() / 2.0
    weight[1:] += learning_rate * data.T.dot(errors)
    weight[0] += learning_rate * errors.sum()

    return cost


if __name__ == "__main__":
    print("\nCheck update_weight(data, label):\n")

    iteration = 1000    # 반복할 횟수
    cost_period = 100   # 비용을 저장하는 주기
    costs = list()      # 비용을 저장하는 리스트

    for cnt in range(iteration):
        cost = update_weight(data, label)
        if cnt % cost_period == 0:
            costs.append(cost)

    print(f"iteration: {iteration}    cost_period: {cost_period}")
    print("weight = [" + ' '.join(["{:0.4f}".format(num) for num in weight]) + "]")
    print("costs  = [" + ' '.join(["{:0.4f}".format(num) for num in costs])  + "]")

    print("\nprediction test:")
    for idx, d in enumerate(data):
        l = predict(d, weight)
        print("data: [" + ' '.join(["{:0.4f}".format(num) for num in d]) + "]   label: {:2d}".format(l) +
              "    passed" if l == label[idx] else "    failed")



#################
#   Problem 6   #
################# 
# 
# 위에서 구현한 모든 함수들을 이용하여 이진 분류기 모델을 구현하라.
# (단, shuffle() 함수는 제외) 
# 
# 위에서 구현한 함수를 그대로 사용하지 않고 추가적으로 최적화해도 좋다.
# self.fit(data, label) 메소드를 호출하면 모델의 학습이 완료되도록 정의하라.
# data는 원본 데이터 샘플, label은 해당 샘플에 대한 정답 레이블 벡터이다. 
# (5번 문제의 확인 코드를 참고해도 됨)
# 그 외 모든 메소드의 이름은 앞서 구한 것과 동일하게 한다.
# 
# 모델의 확인을 위해 'self.costs_'라는 리스트에 학습의 매 회차마다의 비용값들을 저장하라.  


# 답안:

class Adaline(object):
    def __init__(self, eta=0.01, n_iter=1000, random_state=1):
        self.eta = eta                    # 학습률
        self.n_iter = n_iter              # 반복 횟수
        self.random_state = random_state  # 랜덤 시드값
        self.weight_ = None
        self.costs_ = None


    def fit(self, data, label):
        self.weight_ = self.weight_initializer(data)
        self.costs_ = list()

        for _ in range(self.n_iter):
            self.costs_.append(self.update_weight(data, label))
        
        return self


    def weight_initializer(self, data):
        rgen = np.random.RandomState(seed=self.random_state)
        return rgen.normal(loc=0.0, scale=0.01, size=data.shape[1]+1)


    def net_input(self, data, weight):
        return np.dot(data, weight[1:]) + weight[0]


    def activation(self, net_input):
        return net_input


    def update_weight(self, data, label):
        output = self.activation(self.net_input(data, self.weight_))
        errors = label - output
        cost = (errors ** 2).sum() / 2.0
        self.weight_[1:] += self.eta * data.T.dot(errors)
        self.weight_[0] += self.eta * errors.sum()

        return cost

    
    def predict(self, data):
        return np.where(self.activation(self.net_input(data, self.weight_)) >= 0, 1, -1)

    
    def plot_costs(self, ax=plt):
        if self.costs_ is None:
            return
            
        ax.plot(range(1, len(self.costs_) + 1), self.costs_, marker='o')


if __name__ == "__main__":
    # 붓꽃 데이터셋 학습 알고리즘
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    
    # choosing only 'setosa' and 'versicolor'
    y = df.iloc[0:100, 4].values   
    y = np.where(y == 'Iris-setosa', -1, 1)  # labeling 'setosa' as -1 and 'versicolor' as 1

    # distract features of 'setosa' and 'versicolor'
    X = df.iloc[0:100, [0, 2]].values  

    # standardization of the samples
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:,1].std()

    # fitting model
    model = Adaline(eta=0.01, n_iter=15, random_state=1)
    model.fit(X_std, y)

    # plotting result
    adaline_result = plt.figure(figsize=(12, 5))
    ax1 = adaline_result.add_subplot(1, 2, 1)
    ax2 = adaline_result.add_subplot(1, 2, 2)

    model.plot_costs(ax=ax1)
    ax1.set_title('Costs per Epoch')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('costs')

    plot_decision_regions(X_std, y, model, resolution=0.02, ax=ax2)
    ax2.set_title('Decision Region')
    ax2.set_xlabel('sepal length [cm]')
    ax2.set_ylabel('petal length [cm]')
    ax2.legend(loc='upper left')

    plt.show()
