"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import cifar10_mask as cifar10
import cifar10_mask_input as cifar10_input

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import LocalResponseNormalization2DLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, InputLayer, DimshuffleLayer, BatchNormLayer
from lasagne.regularization import regularize_layer_params_weighted, l2

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

batch_size = 100
train_dir = "./cifar10_theano_train_adam"
max_steps = 100000
log_device_placement = False
validation=False
validation_model = ""


def train(image_with_bkg, target_mask_file_name, target_decluttered_file_name):
    """Train CIFAR-10 for a number of steps."""

    image_input_var = T.tensor4('original_inputs')

    cnn_model = cifar10.build_cnn(image_input_var)

    model_output_eval = lasagne.layers.get_output(cnn_model, deterministic=True)

    params = lasagne.layers.get_all_params(cnn_model, trainable=True)

    val_fn = theano.function([image_input_var], model_output_eval)

    if os.path.isfile(os.path.join(train_dir, 'latest_model_mask_5x5.txt')):
    #if os.path.isfile(os.path.join(train_dir, 'latest_model_mask_1x1.txt')):
    #if os.path.isfile(os.path.join(train_dir, 'latest_model_mask_5x3x3xfully.txt')):
        weight_file = ""
        with open(os.path.join(train_dir, 'latest_model_mask_5x5.txt'), 'r') as checkpoint_file:
        #with open(os.path.join(train_dir, 'latest_model_mask_1x1.txt'), 'r') as checkpoint_file:
        #with open(os.path.join(train_dir, 'latest_model_mask_5x3x3xfully.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        for i in range(model_weights.shape[0]):
            print(model_weights[i].shape)
        print("==========================")
        current_weights = lasagne.layers.get_all_param_values(cnn_model)
        for i in range(len(current_weights)):
            print(current_weights[i].shape)
        lasagne.layers.set_all_param_values(cnn_model, model_weights)
    else:
        print("Weights not found")
        sys.exit()


    image_size = image_with_bkg.shape[0]
    batch_size = 100
    image_masks = []
    decluttered_images = []
    threshold = 0.25

    for i in range(image_size // batch_size + 1):
        test_image = image_with_bkg[i * batch_size : min(i * batch_size + batch_size, image_size)]
        if test_image.shape[0] == 0:
            break
        predicted_target = val_fn(np.rollaxis(test_image, 3, 1))
        image_masks.append(predicted_target > threshold)
        current_image_mask = image_masks[-1].reshape(-1, 32, 32, 1)
        current_image_mask = np.repeat(current_image_mask, 3, axis = 3)
        bgcolor = np.array([255, 255, 255]).reshape(1, 1, 1, 3)
        decluttered_images.append(current_image_mask * test_image + (1 - current_image_mask) * bgcolor)

    image_masks = np.vstack(image_masks)
    decluttered_images = np.vstack(decluttered_images)

    np.save(target_mask_file_name, image_masks)
    np.save(target_decluttered_file_name, decluttered_images)



def main(argv=None):  # pylint: disable=unused-argument
    original_image_with_bkg_file = argv[1]
    target_mask_file_name = argv[2]
    target_decluttered_file_name = argv[3]
    image_with_bkg = np.load(original_image_with_bkg_file)
    train(image_with_bkg, target_mask_file_name, target_decluttered_file_name)


if __name__ == '__main__':

    main(sys.argv)
