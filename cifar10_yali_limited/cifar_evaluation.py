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

import skimage
import skimage.transform

batch_size = 400
train_dir = './cifar10_theano_train_adam'
max_steps = 500000
validation = False
validation_model = ''

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

    image_input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, weight_decay_penalty = cifar10.build_cnn(image_input_var)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=False)

    test_acc = T.mean(T.eq(T.argmax(model_output, axis = 1), target_var),
                      dtype=theano.config.floatX)

    val_fn = theano.function([image_input_var, target_var], [test_acc, test_acc])

    if os.path.isfile(os.path.join(train_dir, 'latest_model.txt')):
        weight_file = ""
        with open(os.path.join(train_dir, 'latest_model.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        lasagne.layers.set_all_param_values(cnn_model, model_weights)


    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_input.load_cifar10()

    start_time = time.time() 

    test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
    rotated_test_image = random_rotated_image(test_image)

    total_acc_count = 0
    total_rotate_acc_count = 0
    total_count = 0


    while(test_image is not None):
        loss_value, acc = val_fn(test_image, test_label)
        loss_value_rotate, acc_rotate = val_fn(rotated_test_image, test_label)
        total_rotate_acc_count += acc_rotate * rotated_test_image.shape[0]
        total_acc_count += acc * test_image.shape[0]
        total_count += test_image.shape[0]
        test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
        if test_image is not None:
            rotated_test_image = random_rotated_image(test_image)



    print("Final Accuracy: %.4f" % (float(total_acc_count / total_count)))
    print("Final Rorated Accuracy: %.4f" % (float(total_rotate_acc_count / total_count)))

    print(total_count)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    main()
