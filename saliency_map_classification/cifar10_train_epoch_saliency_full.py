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
import amitgroup as ag
import amitgroup.plot as gr
from scipy import misc
from lasagne.layers import LocalResponseNormalization2DLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, InputLayer, DimshuffleLayer, BatchNormLayer
from lasagne.regularization import regularize_layer_params_weighted, l2


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

batch_size = 100
train_dir = "./cifar10_theano_train_adam"
max_steps = 300
log_device_placement = False
validation=False
validation_model = ""


def train(training_num):

    """Train CIFAR-10 for a number of steps."""

    image_input_var = T.tensor4('original_inputs')
    target_var = T.imatrix('target_mask')

    cnn_model = cifar10.build_cnn(image_input_var)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=False)

    model_output_eval = lasagne.layers.get_output(cnn_model, deterministic=True)

    model_output = T.clip(model_output, 0.00001, 0.99999)

    all_loss = -T.mean(T.log(model_output) * target_var + T.log(1-model_output) * (1 - target_var), axis = 1)

    loss = T.mean(all_loss)
    params = lasagne.layers.get_all_params(cnn_model, trainable=True)[4:]

    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.00001, momentum=0.9)

    train_fn = theano.function([image_input_var, target_var], loss, updates=updates)

    val_fn = theano.function([image_input_var, target_var], [loss, model_output_eval])

    if 1:
        if os.path.isfile(os.path.join(train_dir, 'latest_model_test.txt')):
            weight_file = ""
            with open(os.path.join(train_dir, 'latest_model_test.txt'), 'r') as checkpoint_file:
                weight_file = checkpoint_file.read().replace('\n', '')
            print("Loading from: ", weight_file)
            model_weights = np.load(weight_file)
            current_weights = lasagne.layers.get_all_param_values(cnn_model)
            for i in range(len(model_weights)):
                print(model_weights[i].shape)
            #final_weights = [model_weights[i] for i in range(len(model_weights) - 2)] + [current_weights[-6], current_weights[-5], current_weights[-4], current_weights[-3], current_weights[-2], current_weights[-1]]
            #final_weights = [model_weights[i] for i in range(len(model_weights) - 2)] + [current_weights[-4], current_weights[-3], current_weights[-2], current_weights[-1]]
            final_weights = [model_weights[i] for i in range(len(current_weights) - 2)] + [current_weights[-2], current_weights[-1]]
            #final_weights = [current_weights[-2], current_weights[-1]]
            lasagne.layers.set_all_param_values(cnn_model, final_weights)
    if 0:
        if os.path.isfile(os.path.join(train_dir, 'latest_model_mask.txt')):
            weight_file = ""
            with open(os.path.join(train_dir, 'latest_model_mask.txt'), 'r') as checkpoint_file:
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


    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_input.load_cifar10(training_num)


    for epoch in range(max_steps):
        start_time = time.time()

        if epoch % 10 == 0:

            test_image, test_target = cifar10_data.test.next_eval_batch(batch_size)

            print("Start Evaluating %d" % epoch)

            total_loss_value = 0
            total_count = 0
            tmp_predicted = None
            tmp_target = None
            while(test_image is not None):
                loss_value, predicted_target = val_fn(test_image, test_target)
                tmp_predicted = predicted_target
                tmp_target = test_target
                total_count += test_image.shape[0]
                total_loss_value += loss_value * test_image.shape[0]
                test_image, test_target = cifar10_data.test.next_eval_batch(batch_size)

            print("Final Test Loss: %.4f" % (float(total_loss_value / total_count)))

        print("Start To Train")
        train_image, train_target, start = cifar10_data.train.next_batch(batch_size)
        end_time_1 = time.time() - start_time
        step = 1
        total_loss_value = 0
        total_count = 0
        start = 1
        while(start != 0):
            loss_value = train_fn(train_image, train_target)
            train_image, train_target, start = cifar10_data.train.next_batch(batch_size)
            step += 1
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            total_loss_value += loss_value * train_image.shape[0]
            total_count += train_image.shape[0]

        print("Epoch Stop, loss_averge", float(total_loss_value) / float(total_count))
        duration = time.time() - start_time
        print("Duration is", duration)

        if epoch % 40 == 0 or (epoch + 1) == max_steps:
            checkpoint_path = os.path.join(train_dir, 'model_mask_step%d_mask_5x5.npy' % epoch)
            weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
            np.save(checkpoint_path, weightsOfParams)
            latest_model_path = os.path.join(train_dir, 'latest_model_mask_5x5.txt')
            try:
                os.remove(latest_model_path)
            except OSError:
                pass
            latest_model_file = open(latest_model_path, "w")
            latest_model_file.write(checkpoint_path)
            latest_model_file.close()
        if epoch % 10 == 0:
            show_image = np.zeros((7 * 32, 7 * 32))
            for m in range(7):
                for n in range(7):
                    show_image[m * 32: m * 32 + 32, n * 32:n * 32 + 32] = tmp_predicted[m * 7 + n].reshape(32, 32)
            misc.imsave("./cifar10_theano_train_adam/predicted_mask/predicted_mask%d.png"%(epoch * 10), show_image)
            """
            gr.images(tmp_predicted[:50].reshape(50, 32, 32))
            gr.images(tmp_target[:50].reshape(50, 32, 32))
            np.save("./cifar10_theano_train_adam/predicted_mask_step%d.npy" %epoch, tmp_predicted[:50].reshape(50, 32, 32))
            """


def main(argv=None):  # pylint: disable=unused-argument
    training_num = int(sys.argv[1])
    train(training_num)


if __name__ == '__main__':
    main()
