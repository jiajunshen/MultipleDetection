import lasagne
import numpy as np
import theano.tensor as T
from PIL import Image
import theano
import os, sys, gzip
from six.moves import urllib
import tarfile
import pickle
import cifar10_input
import lasagne
from lasagne.layers import LocalResponseNormalization2DLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, InputLayer, DimshuffleLayer, BatchNormLayer
from lasagne.regularization import regularize_layer_params_weighted, l2
#from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer

IMAGE_SIZE = cifar10_input.IMAGE_SIZE


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def build_cnn(input_var=None):
    """Build the CIFAR-10 model.

    Args:
    images: Images returned from distorted_inputs() or inputs().

    Returns:
    Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    input_layer = InputLayer((None, 3, IMAGE_SIZE, IMAGE_SIZE), input_var=input_var)

    norm0 = BatchNormLayer(input_layer)

    # conv1
    conv1 = Conv2DLayer(norm0, num_filters=64, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.0),
                        name="conv1")

    conv1a = Conv2DLayer(conv1, num_filters=64, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.0),
                        name="conv1a")


    norm1 = BatchNormLayer(conv1a)
    pool1 = MaxPool2DLayer(norm1, pool_size=(2, 2), stride=(2, 2), pad=0)

    # pool1


    # conv2
    conv2 = Conv2DLayer(lasagne.layers.dropout(pool1, p = 0.5),
                        num_filters=128, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.1),
                        name='conv2')

    conv2a = Conv2DLayer(conv2,
                        num_filters=128, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.1),
                        name='conv2a')

    norm2 = BatchNormLayer(conv2a)

    pool2 = MaxPool2DLayer(norm2, pool_size=(2, 2), stride=(2, 2), pad=0)

    # norm2

    # pool2


    conv3 = Conv2DLayer(lasagne.layers.dropout(pool2, p = 0.5),
                        num_filters=256, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.1),
                        name='conv3')

    norm3 = BatchNormLayer(conv3)

    pool3 = MaxPool2DLayer(norm3, pool_size=(2, 2), stride=(2, 2), pad=0)


    # fc1
    fc1 = DenseLayer(lasagne.layers.dropout(pool3, p = 0.5),
                     num_units=256,
                     nonlinearity=lasagne.nonlinearities.rectify,
                     W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1),
                     name="fc1")

    # fc3
    softmax_layer = DenseLayer(lasagne.layers.dropout(fc1, p = 0.5),
                               num_units=9,
                               nonlinearity=lasagne.nonlinearities.softmax,
                               W=lasagne.init.GlorotUniform(),
                               b=lasagne.init.Constant(0.0),
                               name="softmax")

    # Weight Decay
    weight_decay_layers = {fc1: 0.0}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return softmax_layer, l2_penalty
