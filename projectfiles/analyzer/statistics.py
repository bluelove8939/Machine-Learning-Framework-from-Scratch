import numpy as np


def classification_accuracy(model, data, label):
    predicted_result = model.predict(data)
    if predicted_result.ndim == 2:
        predicted_result = np.array([vec.argmax() for vec in predicted_result])
    correct = np.count_nonzero(predicted_result == label)
    return correct / label.shape[0]