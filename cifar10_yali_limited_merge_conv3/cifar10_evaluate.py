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
import cifar10
import cifar10_input
import cifar10_merge_input

import theano
import theano.tensor as T
import lasagne

import skimage
import skimage.transform

batch_size = 400

train_dir = './cifar10_theano_train_merge'

max_steps = 100000

load_model = './cifar10_theano_train_adam/model_step1.npy'

def random_rotated_image(images, low = -20, high = 20):
    image_num, image_channel, image_height, image_width = images.shape
    rotated_degree_1 = list(np.random.randint(low = low, high = 5, size = image_num // 2))
    rotated_degree_2 = list(np.random.randint(low = 5, high = high, size = image_num // 2))
    rotated_degree = np.array(rotated_degree_1 + rotated_degree_2)
    np.random.shuffle(rotated_degree)

    rotated_image = [skimage.transform.rotate(np.rollaxis(images[i], 0, 3), rotated_degree[i], mode="reflect") for i in range(image_num)]
    rotated_image = np.rollaxis(np.array(rotated_image), 3, 1)
    return np.array(rotated_image, dtype = np.float32)

def train():
    """Train CIFAR-10 for a number of steps."""
    if os.path.isfile(load_model):
        all_weights = np.load(load_model) 
    else:
        print("Model file does not exist. Exiting....")
        return

    print("Build up the network")
    target_var = T.ivector('targets')
    rotated_image_input_var = T.tensor4('rotated_image_input')

    # cnn_model_val, _, _ = cifar10_merge.build_cnn(rotated_image_input_var)
    cnn_model_val, _ = cifar10.build_cnn(rotated_image_input_var)

    original_model_output_val = lasagne.layers.get_output(cnn_model_val, rotated_image_input_var, deterministic = False)

    lasagne.layers.set_all_param_values(cnn_model_val, all_weights)

    original_model_acc = T.mean(T.eq(T.argmax(original_model_output_val, axis = 1), target_var),
                                dtype=theano.config.floatX)

    val_fn = theano.function(inputs = [rotated_image_input_var, target_var],
                             outputs = [original_model_acc, original_model_acc])

    # Get images and labels for CIFAR-10.

    #cifar10_data = cifar10_merge_input.load_cifar10()
    cifar10_data = cifar10_input.load_cifar10()

    start_time = time.time()

    original_test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)

    total_acc_count = 0
    original_acc_count = 0
    total_acc_count_original_image = 0
    original_acc_count_original_image = 0
    total_count = 0

    print("Start Evaluating")

    while(original_test_image is not None):
        train_acc_original_image, original_acc_original_image = val_fn(original_test_image, test_label)
        total_acc_count_original_image += train_acc_original_image * original_test_image.shape[0]

        # train_acc, original_acc = val_fn(rotated_test_image, test_label)
        # total_acc_count += train_acc * rotated_test_image.shape[0]

        total_count += original_test_image.shape[0]
        original_test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
    
    print("Teacher Network Accuracy on Original Image: %.4f" % (float(total_acc_count_original_image / total_count)))
    print("Teacher Network Accuracy on Rotated Image: %.4f" % (float(total_acc_count / total_count)))
    
    print(total_count)

def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    main()
