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
#from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.regularization import regularize_layer_params_weighted, l2
from rotationMatrixLayer import RotationTransformationLayer

from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer

from dataPreparation import load_data
from repeatLayer import Repeat
from rotationMatrixLayer import RotationTransformationLayer
from selectLayer import SelectLayer
from CNNForMnist_Rotation_Net import rotateImage_batch
from CNNForMnist_Rotation_Net import rotateImage
from collections import OrderedDict

def build_cnn(input_var=None, batch_size = None, class_num=10):

    # Input layer, as usual:
    l_in = lasagne.layers.InputLayer(shape=(batch_size, 1, 40, 40),
                                        input_var=input_var)

    loc_network_list = []
    for i in range(class_num):
        loc_l1 = MaxPool2DLayer(l_in, pool_size=(2, 2))
        loc_l2 = Conv2DLayer(
            loc_l1, num_filters=20, filter_size=(5, 5), W=lasagne.init.HeUniform('relu'), name = "loc_l2_%d" %i)
        loc_l3 = MaxPool2DLayer(loc_l2, pool_size=(2, 2))
        loc_l4 = Conv2DLayer(loc_l3, num_filters=20, filter_size=(5, 5), W=lasagne.init.HeUniform('relu'), name = "loc_l4_%d" %i)
        loc_l5 = lasagne.layers.DenseLayer(
            loc_l4, num_units=50, W=lasagne.init.HeUniform('relu'), name = "loc_l5_%d" %i)
        loc_out = lasagne.layers.DenseLayer(
            loc_l5, num_units=1, W=lasagne.init.Constant(0.0), 
            nonlinearity=lasagne.nonlinearities.identity, name = "loc_out_%d" %i)
        # Transformer network
        l_trans1 = RotationTransformationLayer(l_in, loc_out)
        print "Transformer network output shape: ", l_trans1.output_shape
        loc_network_list.append(l_trans1)
    network_transformed = lasagne.layers.ConcatLayer(loc_network_list, axis = 1)

    network_transformed = lasagne.layers.ReshapeLayer(network_transformed, (-1, 1, 40, 40))

    conv_1 = Conv2DLayer(
        network_transformed, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = MaxPool2DLayer(conv_1, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    conv_2 = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    network = lasagne.layers.MaxPool2DLayer(conv_2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc1, p=.5),
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=10,
            )

    fc2_selected = SelectLayer(fc2, 10)
    # fc2_selected = lasagne.layers.NonlinearityLayer(fc2_selected, nonlinearity=lasagne.nonlinearities.softmax)

    #network_transformed = lasagne.layers.ReshapeLayer(network_transformed, (-1, 10, 10, 40, 40))

    weight_decay_layers = {fc1:0.0, fc2:0.002}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc2_selected, l2_penalty, network_transformed, [conv_1, conv_2, fc1, fc2] 
    
