# coding=utf-8
import numpy as np


def predict(test_images, theta):
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds

def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    count = 0
    for i in range(y.shape[0]):
        if y_pred[i] == y[i]:
            count += 1
    acc = count / y.shape[0]
    return acc