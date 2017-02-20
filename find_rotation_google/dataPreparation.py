import os
import numpy as np
import sys
import time

def load_data(trainingData, trainingLabel, testingData, testingLabel, dataset = "GOOGLE"):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel

    X_train = np.array(np.load(trainingData) / 255.0, dtype = np.float32).reshape(-1, 48, 48, 3)
    X_train = np.rollaxis(X_train, 3, 1)
    Y_train = np.array(np.load(trainingLabel), dtype = np.uint8)
    X_test = np.array(np.load(testingData) / 255.0, dtype = np.float32).reshape(-1, 48, 48, 3)
    X_test = np.rollaxis(X_test, 3, 1)
    Y_test = np.array(np.load(testingLabel), dtype = np.uint8)

    return X_train, Y_train, X_test, Y_test
