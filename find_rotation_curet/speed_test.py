# This file is a editted version of https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
# Use Lasagne for digit recognition using  MNIST dataset.
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
import cv2
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.regularization import regularize_layer_params_weighted, l2
from  lasagne.layers import LocalResponseNormalization2DLayer as LRN
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data
from repeatLayer import Repeat
from rotationMatrixLayer import RotationTransformationLayer
from selectLayer import SelectLayer
from collections import OrderedDict


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], indices[start_idx:start_idx+batchsize]


def extend_images(images, dim = 277):
    extended_images_res = np.pad(images, ((0,0),(0,0),(13,14),(13,14)), mode="wrap")
    return extended_images_res


def main(model='mlp', num_epochs=2000):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class) 
    print("Using all the training data") 
    
    ## Load Data##
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

    
    X_train = extend_images(X_train, 227)
    X_test_all = extend_images(X_test, 227)
    X_test = extend_images(X_test, 227)[:20]

    y_train = y_train
    y_test_all = y_test[:]
    y_test = y_test[:20]


    ## Define Batch Size ##
    batch_size = 10
 
    ## Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)
    input_network = lasagne.layers.InputLayer(shape=(None, 1, 227, 227),
                                        input_var=input_var)

    repeatInput = Repeat(input_network, 61)

    repeatInput = lasagne.layers.ReshapeLayer(repeatInput, (-1, 1, 227, 227))

    rotatedInput = RotationTransformationLayer(repeatInput, batch_size * 61)

    conv1 = Conv2DLayer(
            repeatInput, num_filters=96, filter_size=(11, 11),
            stride=(4,4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    conv1 = LRN(conv1, alpha = 0.0001, beta = 0.75, n = 5)

    # Max-pooling layer of factor 2 in both dimensions:
    conv1 = MaxPool2DLayer(conv1, pool_size=(3, 3), stride=(2,2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    conv2 = Conv2DLayer(
            conv1, num_filters=256, filter_size=(5, 5),
            pad = 2,
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    
    conv2 = LRN(conv2, alpha = 0.0001, beta = 0.75, n = 5)
    
    conv2 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(3, 3), stride = (2, 2))
    

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1  = lasagne.layers.DenseLayer(
            conv2,
            num_units=256,
            W = lasagne.init.Normal(0.01),
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            fc1,
            num_units=4096,
            W = lasagne.init.Normal(0.005),
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    fc3  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc2, p=.5),
            num_units=4096,
            W = lasagne.init.Normal(0.005),
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    fc4  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc3, p=.5),
            num_units=61,
            nonlinearity=lasagne.nonlinearities.identity,
            )

    fc4_selected = SelectLayer(fc4, 61)

    repeatInput_result = lasagne.layers.get_output(repeatInput)
    get_repeat_fn = theano.function([input_var], repeatInput_result)
    rotatedInput_result = lasagne.layers.get_output(rotatedInput)
    get_rotated_fn = theano.function([input_var], rotatedInput_result)
    conv1_result = lasagne.layers.get_output(conv1)
    get_conv1_fn = theano.function([input_var], conv1_result)
    conv2_result = lasagne.layers.get_output(conv2)
    get_conv2_fn = theano.function([input_var], conv2_result)
    fc1_result = lasagne.layers.get_output(fc1)
    get_fc1_fn = theano.function([input_var], fc1_result)
    fc2_result = lasagne.layers.get_output(fc2)
    get_fc2_fn = theano.function([input_var], fc2_result)
    fc3_result = lasagne.layers.get_output(fc3)
    get_fc3_fn = theano.function([input_var], fc3_result)
    fc4_result = lasagne.layers.get_output(fc4)
    get_fc4_fn = theano.function([input_var], fc4_result)
    fc4_selected_result = lasagne.layers.get_output(fc4_selected)
    get_fc4_selected_fn = theano.function([input_var], fc4_selected_result)

    
    i = 0
    for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
        inputs, targets, index = batch
        inputs = inputs.reshape(batch_size, 1, 227, 227)
        time_start = time.time()

        if i == 0:
            result_repeat = get_repeat_fn(inputs)
            time_repeat = time.time() - time_start
            print(time_repeat)
        if i == 1:
            result_rotation = get_rotated_fn(inputs)
            time_rotation = time.time() - time_start
            print(time_rotation)
        if i == 2:
            result_conv1 = get_conv1_fn(inputs)
            time_conv1 = time.time() - time_start
            print(time_conv1)
        if i == 3:
            result_conv2 = get_conv2_fn(inputs)
            time_conv2 = time.time() - time_start
            print(time_conv2)
        if i == 4:
            result_fc1 = get_fc1_fn(inputs)
            time_fc1 = time.time() - time_start
            print(time_fc1)
        if i == 5:
            result_fc2 = get_fc2_fn(inputs)
            time_fc2 = time.time() - time_start
            print(time_fc2)
        if i == 6:
            result_fc3 = get_fc3_fn(inputs)
            time_fc3 = time.time() - time_start
            print(time_fc3)
        if i == 7:
            result_fc4 = get_fc4_fn(inputs)
            time_fc4 = time.time() - time_start
            print(time_fc4)
        if i == 8:
            result_fc4_selected = get_fc4_selected_fn(inputs)
            time_fc4_selected = time.time() - time_start
            print(time_fc4_selected)
        i+=1
        if i % 9 == 0:
            print("----------------------------------------")
        i = i % 9
        """
        print("time_repeat", time_repeat)
        print("time_rotation", time_rotation - time_repeat)
        print("time conv1", time_conv1 - time_rotation)
        print("time conv2", time_conv2 - time_conv1)
        print("time fc1", time_fc1 - time_conv2)
        print("time fc2", time_fc2 - time_fc1)
        print("time fc3", time_fc3 - time_fc2)
        print("time fc4", time_fc4 - time_fc3)
        print("time fc4 selected", time_fc4_selected - time_fc4)
        print("time_repeat", time_repeat)
        print("time_rotation", time_rotation)
        print("time conv1", time_conv1)
        print("time conv2", time_conv2)
        print("time fc1", time_fc1)
        print("time fc2", time_fc2)
        print("time fc3", time_fc3)
        print("time fc4", time_fc4)
        print("time fc4 selected", time_fc4_selected)
        """

    print (time.time() - start_time)



if __name__ == '__main__':
    main()
