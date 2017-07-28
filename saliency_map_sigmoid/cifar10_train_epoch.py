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

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import cifar10
import cifar10_input

import theano
import theano.tensor as T
import lasagne


#tf.app.flags.DEFINE_integer('batch_size', 400,
#                            """size of batches.""")
batch_size = 400
#tf.app.flags.DEFINE_string('train_dir', './cifar10_theano_train_adam',
#                           """Directory where to write event logs """
#                           """and checkpoint.""")
train_dir = "./cifar10_theano_train_adam"
max_steps = 100
log_device_placement = False
validation=False
validation_model = ""
#tf.app.flags.DEFINE_integer('max_steps', 500000,
#                            """Number of batches to run.""")
#tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                            """Whether to log device placement.""")
#tf.app.flags.DEFINE_boolean('validation', False,
#                            """Whether to validate the model.""")
#tf.app.flags.DEFINE_string('validation_model', '',
#                            """The model file to validate.""")


def train():
    """Train CIFAR-10 for a number of steps."""

    image_input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, weight_decay_penalty = cifar10.build_cnn(image_input_var)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=False)

    cross_entropy_loss = lasagne.objectives.categorical_crossentropy(model_output, target_var)

    cross_entropy_loss_mean = cross_entropy_loss.mean()

    loss = cross_entropy_loss_mean + weight_decay_penalty

    params = lasagne.layers.get_all_params(cnn_model, trainable=True)

    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

    train_fn = theano.function([image_input_var, target_var], [loss, cross_entropy_loss_mean, weight_decay_penalty], updates=updates)

    test_acc = T.mean(T.eq(T.argmax(model_output, axis = 1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([image_input_var, target_var], [loss, test_acc])

    if 1: 
        if os.path.isfile(os.path.join(train_dir, 'latest_model_rectify.txt')):
            weight_file = ""
            with open(os.path.join(train_dir, 'latest_model_rectify.txt'), 'r') as checkpoint_file:
                weight_file = checkpoint_file.read().replace('\n', '')
            print("Loading from: ", weight_file)
            model_weights = np.load(weight_file)
            current_weights = lasagne.layers.get_all_param_values(cnn_model)
            for i in range(len(model_weights)):
                print(model_weights[i].shape)
            final_weights = [model_weights[i] for i in range(len(model_weights) - 2)] + [current_weights[-2], current_weights[-1]]
            lasagne.layers.set_all_param_values(cnn_model, final_weights)


    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_input.load_cifar10()


    for epoch in range(max_steps):
        
        start_time = time.time() 

        test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
        total_acc_count = 0
        total_count = 0

        print("Start Evaluating %d" % epoch)

        while(test_image is not None):
            loss_value, acc = val_fn(test_image, test_label)
            total_acc_count += acc * test_image.shape[0]
            total_count += test_image.shape[0]
            test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)

        print("Final Accuracy: %.4f" % (float(total_acc_count / total_count)))

        print("Start To Train")
        train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
        end_time_1 = time.time() - start_time
        step = 1
        loss_total = 0
        while(start != 0):
            loss_value, cross_entropy_value, weight_decay_value = train_fn(train_image, train_label)
            train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
            step += 1
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            loss_total += loss_value

        print("Epoch Stop, loss_averge", float(loss_total) / float(step))
        duration = time.time() - start_time
        print("Duration is", duration)
        
        if epoch % 100 == 0 or (epoch + 1) == max_steps:
            checkpoint_path = os.path.join(train_dir, 'model_step%d.npy' % epoch)
            weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
            np.save(checkpoint_path, weightsOfParams)
            latest_model_path = os.path.join(train_dir, 'latest_model.txt')
            try:
                os.remove(latest_model_path)
            except OSError:
                pass
            latest_model_file = open(latest_model_path, "w")
            latest_model_file.write(checkpoint_path)
            latest_model_file.close()

def validate():
    """Train CIFAR-10 for a number of steps."""
    if os.path.isfile(validation_model):
        all_weights = np.load(validation_model)
    else:
        print("Model file does not exist. Exiting....")
        return

    print("Build up the network")
    image_input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, weight_decay_penalty = cifar10.build_cnn(image_input_var)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=True)

    all_layers = lasagne.layers.get_all_layers(cnn_model)

    print(lasagne.layers.get_output_shape(all_layers))

    cross_entropy_loss = lasagne.objectives.categorical_crossentropy(model_output, target_var)

    cross_entropy_loss_mean = cross_entropy_loss.mean()

    loss = cross_entropy_loss_mean + weight_decay_penalty

    print("Load Model weights")

    lasagne.layers.set_all_param_values(cnn_model, all_weights)

    test_acc = T.mean(T.eq(T.argmax(model_output, axis = 1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([image_input_var, target_var], [loss, test_acc])


    # Get images and labels for CIFAR-10.

    print("Loading Data")

    cifar10_data = cifar10_input.load_cifar10()

    test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
    total_acc_count = 0
    total_count = 0

    print("Start Evaluating")

    while(test_image is not None):
        loss_value, acc = val_fn(test_image, test_label)
        total_acc_count += acc * test_image.shape[0]
        total_count += test_image.shape[0]
        print(loss_value)
        test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)

    print("Final Accuracy: %.4f" % (float(total_acc_count / total_count)))
    print("Total:", total_count)
    print("correct: ", total_acc_count)

def main(argv=None):  # pylint: disable=unused-argument
    #validate()
    train()


if __name__ == '__main__':
    main()
