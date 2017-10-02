import os
import numpy as np
import sys
import time
from scipy import misc

def load_data(trainingData, trainingLabel,
              testingData, testingLabel,
              resize = False, standardize = True, size = 30, dataset = "SHAPENET"):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel

    X_train = np.array(np.load(trainingData),
                       dtype = np.float32) / 255.0
    Y_train = np.array(np.load(trainingLabel),
                       dtype = np.uint8)
    X_test = np.array(np.load(testingData),
                      dtype = np.float32) / 255.0
    Y_test = np.array(np.load(testingLabel),
                      dtype = np.uint8)

    print("resizing....")
    if resize:
        X_train = np.array([misc.imresize(X_train[i],
                                          size = (size, size, 3)) /255.0
                            for i in range(X_train.shape[0])], dtype=np.float32)
        X_test = np.array([misc.imresize(X_test[i],
                                         size = (size, size, 3)) /255.0
                           for i in range(X_test.shape[0])], dtype=np.float32)
        #np.save(trainingData + "_100.npy", X_train)
        #np.save(testingData + "_100.npy", X_test)

    X_train = np.rollaxis(X_train, 3, 1)
    X_test = np.rollaxis(X_test, 3, 1)

    if standardize:
        X_train = per_image_standardization(X_train)
        X_test = per_image_standardization(X_test)

    return X_train, Y_train, X_test, Y_test

def per_image_standardization(images):
    image_num, image_channel, image_height, image_width = images.shape
    standardized_image = np.array([(images[i] - np.mean(images[i])) / \
                                   max(np.std(images[i]), 1.0 / np.sqrt(image_channel * \
                                                                        image_height * \
                                                                        image_width)) \
                                   for i in range(image_num)])
    return standardized_image
