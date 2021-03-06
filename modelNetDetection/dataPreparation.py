import os
import numpy as np
import sys
import time


def load_data(trainingData, trainingLabel, testingData, testingLabel, dataset="MODELNET40"):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel

    X_train = np.array(np.load(trainingData), dtype=np.float32).reshape(-1, 32, 32, 32)
    Y_train = np.array(np.load(trainingLabel), dtype=np.uint8)
    X_test = np.array(np.load(testingData), dtype=np.float32).reshape(-1, 32, 32, 32)
    Y_test = np.array(np.load(testingLabel), dtype=np.uint8)

    return X_train, Y_train, X_test, Y_test


def load_data_3D(trainingData, trainingLabel, testingData, testingLabel, dataset="MNIST_3D"):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel

    X_train = np.array(np.load(trainingData), dtype=np.float32).reshape(-1, 1, 40, 40, 40)
    Y_train = np.array(np.load(trainingLabel), dtype=np.uint8)
    X_test = np.array(np.load(testingData), dtype=np.float32).reshape(-1, 1, 40, 40, 40)
    Y_test = np.array(np.load(testingLabel), dtype=np.uint8)

    return X_train, Y_train, X_test, Y_test
