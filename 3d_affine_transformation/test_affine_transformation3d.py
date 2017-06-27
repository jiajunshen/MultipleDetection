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
from collections import OrderedDict
from affineTransformation3D import AffineTransformation3DLayer
import h5py

def build_cnn(input_var=None, batch_size = None):
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 16, 16, 16),
                                        input_var=input_var)
    network = AffineTransformation3DLayer(network, batch_size)

    return network

def main():
    hf = h5py.File("/hdd/Documents/Data/3D-MNIST/full_dataset_vectors.h5", "r")
    X_train = hf["X_train"][0].reshape(1, 1, 16, 16, 16)
    X_train = np.rollaxis(X_train, 2, 5)
    X_train = np.array(X_train, dtype = np.float32)
    input_var = T.tensor5('inputs')
    network = build_cnn(input_var, 1)
    network_output = lasagne.layers.get_output(network)

    get_rotated = theano.function([input_var], network_output)
    image_result = get_rotated(X_train)
    import amitgroup as ag
    import amitgroup.plot as gr
    gr.images(np.mean(image_result, axis = 4)[0,0])
    return





if __name__ == '__main__':
    main()
