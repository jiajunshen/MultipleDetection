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

import cifar10_merge
import cifar10_evaluate_input

import theano
import theano.tensor as T
import lasagne

batch_size = 400

train_dir = './cifar10_theano_train_merge'

max_steps = 100000

load_model = './cifar10_theano_train_adam/model_step25.npy'

def train():
    """Train CIFAR-10 for a number of steps."""
    if os.path.isfile(load_model):
        all_weights = np.load(load_model) 
    else:
        print("Model file does not exist. Exiting....")
        return

    print("Build up the network")


    # Two different types of input
    rotated_image_input_var = T.tensor4('rotated_image_input')
    target_var = T.ivector('targets')
    # Build the student network
    
    rotated_cnn_model, rotated_model_mid, rotated_weight_penalty = \
        cifar10_merge.build_cnn(rotated_image_input_var)
    
    # Get the model output of the studenet network.
    rotated_model_output = lasagne.layers.get_output(rotated_cnn_model, rotated_image_input_var, deterministic = True)

    
    if os.path.isfile(os.path.join(train_dir, 'latest_model.txt')):
        weight_file = ""
        with open(os.path.join(train_dir, 'latest_model.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        lasagne.layers.set_all_param_values(rotated_cnn_model, model_weights)

        train_acc = T.mean(T.eq(T.argmax(rotated_model_output, axis = 1), target_var),
                           dtype=theano.config.floatX)

        # Return the accuracy for teacher network and student network, respectively
        val_fn = theano.function(inputs = [rotated_image_input_var, target_var],
                                 outputs = [train_acc])

        cifar10_data_test = cifar10_evaluate_input.load_cifar10_test()
        cifar10_data_rotated_test = cifar10_evaluate_input.load_cifar10_rotated_test()

        
        print("Start Evaluating")
        total_s_net_for_original = 0
        total_s_net_for_rotation = 0

        original_test_image, test_label = cifar10_data_test.test.next_eval_batch(batch_size)
        while(original_test_image is not None):
            s_net_for_original = val_fn(original_test_image, test_label)
            total_s_net_for_original += s_net_for_original * original_test_image.shape[0]
            original_test_image, rotated_test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
        
        print("Student Network Accuracy on Original Image: %.4f" % (float(total_s_net_for_original / 10000.0)))




        rotated_test_image, test_label = cifar10_data_rotated_test.test.next_eval_batch(batch_size)

        while(rotated_test_image is not None):
            s_net_for_rotated = val_fn(rotated_test_image, test_label)
            total_s_net_for_rotation += s_net_for_rotated * rotated_test_image.shape[0]
            original_test_image, rotated_test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
        
        print("Student Network Accuracy on Rotated Image: %.4f" % (float(total_s_net_for_rotation / 10000.0)))


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    main()
