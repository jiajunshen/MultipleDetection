import os
import numpy as np
import sys
import time

def load_data(trainingData, trainingLabel, testingData, testingLabel, dataset = "MNIST", W = 28):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel

    X_train = np.array(np.load(trainingData), dtype = np.float32).reshape(-1, 1, W, W)
    Y_train = np.array(np.load(trainingLabel), dtype = np.uint8)
    X_test = np.array(np.load(testingData), dtype = np.float32).reshape(-1, 1, W, W)
    Y_test = np.array(np.load(testingLabel), dtype = np.uint8)

    return X_train, Y_train, X_test, Y_test


def load_data_digit_clutter(trainingData, trainingLabel, testingData, testingLabel, dataset="MNIST", W = 60):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel




    X_train = np.array(create_sequence_data(np.load(trainingData)), dtype = np.float32).reshape(-1, 1, W, W)
    Y_train = np.array(np.load(trainingLabel), dtype = np.uint8)
    X_test = np.array(create_sequence_data(np.load(testingData)), dtype = np.float32).reshape(-1, 1, W, W)
    Y_test = np.array(np.load(testingLabel), dtype = np.uint8)

    return X_train, Y_train, X_test, Y_test

def get_bbox(spatial_data):
    x_value_index = np.where(np.sum(spatial_data, axis = 1) != 0)[0]
    y_value_index = np.where(np.sum(spatial_data, axis = 0) != 0)[0]
    bbox_x_left = x_value_index[0]
    bbox_x_right = x_value_index[-1]
    bbox_y_left = y_value_index[0]
    bbox_y_right = y_value_index[-1]
    return bbox_x_left, bbox_x_right, bbox_y_left, bbox_y_right

def create_sequence_data(dataset, W = 40):
    num_data, _, w, h = dataset.shape
    sequence_data = np.zeros((num_data, 1, W, W))
    digit_margin = 4 # This is a pre-set value for the margin between the digits
    for i in range(num_data):
        current_image = np.zeros((1, W, W))
        cDigit_bbox = get_bbox(dataset[i, 0])
        center_digit_width = cDigit_bbox[3] - cDigit_bbox[2] + 1

        #Start to deal with the left and the right digit
        left_digit_index = np.random.randint(num_data)
        lDigit_bbox = get_bbox(dataset[left_digit_index, 0])
        right_digit_index = np.random.randint(num_data)
        rDigit_bbox = get_bbox(dataset[right_digit_index, 0])

        # Left/Right Margin to the center digit
        margin_y = (W - center_digit_width) // 2
        current_image[:,:,margin_y:margin_y + center_digit_width] = \
            dataset[i, :, :, cDigit_bbox[2]:cDigit_bbox[3]]

        # Fill in the left digit and the right digit (cropped)
        leftDigitWidth = min(lDigit_bbox[3] - lDigit_bbox[2] + 1, margin_y - 4)
        rightDigitWidth = min(rDigit_bbox[3] - rDigit_bbox[2] + 1, margin_y - 4)
        current_image[:, :, margin_y - 4 - leftDigitWidth : margin_y - 4] = \
            dataset[left_digit_index, :, :,
                    lDigit_bbox[3] - leftDigitWidth : lDigit_bbox[3]]
        current_image[:, :, \
            margin_y + center_digit_width + 4 :
            margin_y + center_digit_width + rightDigitWidth + 4] = \
                dataset[right_digit_index, :, :, \
                    rDigit_bbox[3] - rightDigitWidth : rDigit_bbox[3]]
        sequence_data[i] = current_image
    return sequence_data
