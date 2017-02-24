import os
import numpy as np
import sys
import time
import skimage
from skimage import transform

def load_data(trainingData, trainingLabel, testingData, testingLabel, dataset = "GOOGLE", dataType = "train", downScale = False):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel

    if dataType == "train":
        imgSize = 48
    else:
        imgSize = 68


    X_train = np.array(np.load(trainingData) / 255.0, dtype = np.float32).reshape(-1, imgSize, imgSize, 3)

    if downScale and dataType == "test":
        X_train = X_train[:, 10:58, 10:58, :]
        X_train = np.array([skimage.transform.pyramid_reduce(X_train[i]) for i in range(X_train.shape[0])], dtype = np.float32)

    import scipy
    #X_train = scipy.misc.imresize(X_train, 0.5)
    X_train = np.rollaxis(X_train, 3, 1)
    Y_train = np.array(np.load(trainingLabel), dtype = np.uint8)
    X_test = np.array(np.load(testingData) / 255.0, dtype = np.float32).reshape(-1, imgSize, imgSize, 3)
    
    if downScale and dataType == "test":
        X_test = X_test[:, 10:58, 10:58, :]
        X_test = np.array([skimage.transform.pyramid_reduce(X_test[i]) for i in range(X_test.shape[0])], dtype = np.float32)

    #X_test = scipy.misc.imresize(X_test, 0.5)
    X_test = np.rollaxis(X_test, 3, 1)
    Y_test = np.array(np.load(testingLabel), dtype = np.uint8)

    print(np.max(X_train), np.max(X_test))

    return X_train, Y_train, X_test, Y_test
