# This file is a editted version of https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
# Use Lasagne for digit recognition using  MNIST dataset.
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
#from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data


def build_cnn(input_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 100, 100),
                                        input_var=input_var)
    #network = lasagne.layers.BatchNormLayer(network)
    network_1 = Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(), pad=(3-1)//2)

    network_2 = Conv2DLayer(
            network_1, num_filters=32, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(), pad=(3-1)//2)

    network_3 = MaxPool2DLayer(network_2, pool_size=(2, 2))


    network_4 = Conv2DLayer(
            network_3, num_filters=64, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(), pad=(3-1)//2)

    network_5 = Conv2DLayer(
            network_4, num_filters=64, filter_size=(3, 3),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(), pad=(3-1)//2)

    network_6 = MaxPool2DLayer(network_5, pool_size=(2, 2))

    network_7 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network_6, p=.5),
            #network,
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    network_8 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network_7, p=.5),
            #network,
            num_units=7,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network_1, network_2, network_3, network_4, \
           network_5, network_6, network_7, network_8



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(model='mlp', num_epochs=50):
    # Load the dataset
    print("Loading data...")

    X_train, y_train, X_test, y_test = load_data("/X_plain_train_rotation_100.npy",
                                                 "/Y_train_rotation.npy",
                                                 "/X_plain_test_100.npy",
                                                 "/Y_test.npy",
                                                 resize=False,
                                                 size = 100)


    print(X_train.shape, X_test.shape)

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    predictions = []
    for i in range(8):
        predictions.append(lasagne.layers.get_output(network[i], deterministic=True))
    
    #weightsOfParams = np.load("../data/plain_rotation_network_withoutBatchNorm.npy")
    #lasagne.layers.set_all_param_values(network[-1], weightsOfParams)
    
    train_fn = theano.function([input_var], predictions)

    filter_1layer = np.zeros((32, 3, 3, 3))
    filter_2layer = np.zeros((32, 5, 5, 3))
    filter_3layer = np.zeros((32, 6, 6, 3))
    filter_list = [filter_1layer, filter_2layer, filter_3layer]
    filter_count = [np.zeros(32), np.zeros(32), np.zeros(32)]

    # Finally, launch the training loop.
    print("Starting evaluating")

    for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
        inputs, targets = batch
        predictions = train_fn(inputs)
        for i in range(3):
            print("-----------")
            print(predictions[i].shape)
            if i < 2:
                filter_size = filter_list[i].shape[1]
                feature_size = predictions[i].shape[2]
                padded_image = extend_image(inputs, size = 100 + (filter_size-1))
                for j in range(100):
                        for p in range(feature_size):
                            for q in range(feature_size):
                                filter_count[i] += predictions[i][j, :, p, q]
                                image_region = np.rollaxis(padded_image[j, :, p : p + filter_size, q  : q + filter_size], 0, 3)
                                image_region = np.repeat(image_region, 32)
                                image_region = image_region * predictions[i][j,:,p,q].reshape(32, 1, 1, 1)
                                filter_list[i] += image_region
            else:
                filter_size = 6
                feature_size = 50
                padded_image = extend_image(inputs, size = 104)
                for j in range(100):
                    for p in range(feature_size):
                        for q in range(feature_size):
                                filter_count[i] += predictions[i][j, :, p, q]
                                image_region = np.rollaxis(padded_image[j, :, 2 * p : 2 * p + filter_size, 2 * q : 2 * q + filter_size], 0, 3)
                                image_region = np.repeat(image_region, 32)
                                image_region = image_region * predictions[i][j,:,p,q].reshape(32, 1, 1, 1)
                                filter_list[i] += image_region
                            
                            
def extend_image(inputs, size = 40):
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    margin_size = (40 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images
            

if __name__ == '__main__':
    main()
