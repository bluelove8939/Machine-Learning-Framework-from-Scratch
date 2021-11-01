import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical

from neural_network_model import Sequential
from common.layers        import Dense, Flatten
from common.optimizers    import SGD, AdaGrad, Adam
from common.initializers  import Zeros, HeNormal, RandomNormal
from analyzer.statistics  import classification_accuracy


if __name__ == "__main__":
    # MNIST 손글씨 데이터셋 학습 알고리즘
    # URL: https://keras.io/ko/datasets/#mnist
    (x_train, y_train), (x_test, y_test) = load_data()

    ynames = {}
    for lbl in np.unique(y_train): ynames[lbl] = lbl

    y_train_encoded = to_categorical(y_train)

    x_train, x_test = x_train/255.0, x_test/255.0

    # fitting model
    random_state = None
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    # optimizer = AdaGrad(lr=0.001, epsilon=1e-8)
    # optimizer = SGD(lr=0.001, momentum=0, decay=0)
    bias_initializer   = Zeros()

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(units=512, activation='relu', input_shape=784,
                    kernel_initializer=HeNormal(fan_in=784, random_state=random_state),
                    bias_initializer=bias_initializer))
    model.add(Dense(units=100, activation='relu', input_shape=512,
                    kernel_initializer=HeNormal(fan_in=512, random_state=random_state),
                    bias_initializer=bias_initializer))
    model.add(Dense(units=10, activation='softmax', input_shape=100,
                    kernel_initializer=HeNormal(fan_in=100, random_state=random_state),
                    bias_initializer=bias_initializer))
    model.compile(optimizer=optimizer, loss='cee')

    # model = Sequential()
    # model.add(Flatten(input_shape=(28, 28)))
    # model.add(Dense(units=512, activation='relu', input_shape=784,
    #                 kernel_initializer=RandomNormal(mean=0., stddev=0.01, random_state=random_state),
    #                 bias_initializer=bias_initializer))
    # model.add(Dense(units=100, activation='relu', input_shape=512,
    #                 kernel_initializer=RandomNormal(mean=0., stddev=0.01, random_state=random_state),
    #                 bias_initializer=bias_initializer))
    # model.add(Dense(units=10, activation='softmax', input_shape=100,
    #                 kernel_initializer=RandomNormal(mean=0., stddev=0.01, random_state=random_state),
    #                 bias_initializer=bias_initializer))
    # model.compile(optimizer=optimizer, loss='cee')
    #
    model.fit(x_train, y_train_encoded, batch_size=1000, epochs=200, verbose=1)

    # plotting result
    result = plt.figure(figsize=(6, 5))
    ax1 = result.add_subplot()

    model.plot_costs(ax=ax1, resolution=10)
    ax1.set_title('DNN test(mnist): Costs per Epoch')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('costs')

    # testing
    print(f"accuracy: {classification_accuracy(model, x_test, y_test)*100:.2f}%")

    plt.show()
