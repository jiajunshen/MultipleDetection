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
import tensorflow as tf

import cifar10
import cifar10_input

import theano
import theano.tensor as T
import lasagne

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """size of batches.""")

tf.app.flags.DEFINE_string('train_dir', '/home-nfs/jiajun/Research/MultipleDetection/cifar10_experiment/cifar10_theano_train_adagrad',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 500000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('validation', False,
                            """Whether to validate the model.""")
tf.app.flags.DEFINE_string('validation_model', '',
                            """The model file to validate.""")


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

    updates = lasagne.updates.adagrad(loss, params, learning_rate=0.01)

    train_fn = theano.function([image_input_var, target_var], loss, updates=updates)

    test_acc = T.mean(T.eq(T.argmax(model_output, axis = 1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([image_input_var, target_var], [loss, test_acc])


    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_input.load_cifar10()


    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        train_image, train_label = cifar10_data.train.next_batch(FLAGS.batch_size)
        end_time_1 = time.time() - start_time
        loss_value = train_fn(train_image, train_label)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch); %.3f sec/batch prepration')
            print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch, float(end_time_1)))

        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model_step%d.npy' % step)
            weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
            np.save(checkpoint_path, weightsOfParams)

            # Start to Evaluate

            test_image, test_label = cifar10_data.test.next_eval_batch(FLAGS.batch_size)
            total_acc_count = 0
            total_count = 0

            print("Start Evaluating")

            while(test_image is not None):
                loss_value, acc = val_fn(test_image, test_label)
                total_acc_count += acc * test_image.shape[0]
                total_count += test_image.shape[0]
                test_image, test_label = cifar10_data.test.next_eval_batch(FLAGS.batch_size)

            print("Final Accuracy: %.4f" % (float(total_acc_count / total_count)))

def validate():
    """Train CIFAR-10 for a number of steps."""
    if os.path.isfile(FLAGS.validation_model):
        all_weights = np.load(FLAGS.validation_model)
    else:
        print("Model file does not exist. Exiting....")
        return

    print("Build up the network")
    image_input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, weight_decay_penalty = cifar10.build_cnn(image_input_var)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=False)

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

    test_image, test_label = cifar10_data.test.next_eval_batch(FLAGS.batch_size)
    total_acc_count = 0
    total_count = 0

    print("Start Evaluating")

    while(test_image is not None):
        loss_value, acc = val_fn(test_image, test_label)
        total_acc_count += acc * test_image.shape[0]
        total_count += test_image.shape[0]
        print(loss_value)
        test_image, test_label = cifar10_data.test.next_eval_batch(FLAGS.batch_size)

    print("Final Accuracy: %.4f" % (float(total_acc_count / total_count)))
    print("Total:", total_count)
    print("correct: ", total_acc_count)

def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.validation:
        validate()
    else: 
        train()


if __name__ == '__main__':
    tf.app.run()
