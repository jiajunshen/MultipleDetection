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
import cifar10_merge_input

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
    image_input_var = T.tensor4('original_inputs')
    rotated_image_input_var = T.tensor4('rotated_image_input')
    target_var = T.ivector('targets')

    # Build teacher network
    cnn_model, cnn_mid_output, weight_decay_penalty = cifar10_merge.build_cnn(image_input_var)

    # Get the intermediate layer of the teacher network
    original_model_mid_output = lasagne.layers.get_output(cnn_mid_output, image_input_var, deterministic = True)

    # Get the softmax output of the teacher network.

    original_model_output_val = lasagne.layers.get_output(cnn_model, image_input_var, deterministic = True)
    
    # Build the student network
    
    rotated_cnn_model, rotated_model_mid, rotated_weight_penalty = \
        cifar10_merge.build_cnn(rotated_image_input_var)
    
    # Get the softmax output of the student network. Since it need to be trained on, deterministic = False
    rotated_model_mid_output = lasagne.layers.get_output(rotated_model_mid, rotated_image_input_var, deterministic = False)

    # Get the model output of the studenet network.
    rotated_model_output = lasagne.layers.get_output(rotated_cnn_model, rotated_image_input_var, deterministic = True)

    # Set the weights for the teacher network
    lasagne.layers.set_all_param_values(cnn_model, all_weights)

    # Get the initialized weights below the intermediate layer
    rotated_net_weights_below_mid = lasagne.layers.get_all_param_values(rotated_model_mid)

    # Get the parameter of the student network that needs to be trained.
    rotated_net_training_param = lasagne.layers.get_all_params(rotated_model_mid, trainable=True)

    # Set the weights for the student network
    lasagne.layers.set_all_param_values(rotated_cnn_model, all_weights)

    lasagne.layers.set_all_param_values(rotated_model_mid,
                                         rotated_net_weights_below_mid)
    
    # cross_entropy_loss = lasagne.objectives.categorical_crossentropy(rotated_model_mid_output, target_var)

    # cross_entropy_loss_mean = cross_entropy_loss.mean()

    # L = T.mean(lasagne.objectives.squared_error(original_model_mid_output, rotated_model_mid_output), axis = 1)
    L = lasagne.objectives.squared_error(original_model_mid_output, rotated_model_mid_output).mean()
    # cost = T.mean(L)

    # cost = cross_entropy_loss_mean
    cost = L

    # updates = lasagne.updates.adagrad(cost, rotated_net_training_param, learning_rate=0.1)
    updates = lasagne.updates.adam(cost, rotated_net_training_param, learning_rate=0.001)

    # cross_entropy_loss = lasagne.objectives.categorical_crossentropy(model_output, target_var)

    # cross_entropy_loss_mean = cross_entropy_loss.mean()

    # loss = cross_entropy_loss_mean + weight_decay_penalty


    train_acc = T.mean(T.eq(T.argmax(rotated_model_output, axis = 1), target_var),
                       dtype=theano.config.floatX)

    original_model_acc = T.mean(T.eq(T.argmax(original_model_output_val, axis = 1), target_var),
                                dtype=theano.config.floatX)

    train_fn = theano.function(inputs = [image_input_var, rotated_image_input_var, target_var],
                               outputs = [original_model_mid_output, rotated_model_mid_output, train_acc], updates = updates)

    # Return the accuracy for teacher network and student network, respectively
    val_fn = theano.function(inputs = [image_input_var, rotated_image_input_var, target_var],
                             outputs = [original_model_acc, train_acc])

    if os.path.isfile(os.path.join(train_dir, 'latest_model.txt')):
        weight_file = ""
        with open(os.path.join(train_dir, 'latest_model.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        lasagne.layers.set_all_param_values(rotated_cnn_model, model_weights)

    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_merge_input.load_cifar10()


    for epoch in xrange(max_steps):
        start_time = time.time()

        original_test_image, rotated_test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
        total_t_net_for_original = 0
        total_s_net_for_original = 0
        total_t_net_for_rotation = 0
        total_s_net_for_rotation = 0
        total_count = 0

        print("Start Evaluating")

        while(rotated_test_image is not None):
            t_net_for_original, s_net_for_original = val_fn(original_test_image, original_test_image, test_label)
            total_t_net_for_original += t_net_for_original * original_test_image.shape[0]
            total_s_net_for_original += s_net_for_original * original_test_image.shape[0]

            t_net_for_rotated, s_net_for_rotated = val_fn(rotated_test_image, rotated_test_image, test_label)
            total_t_net_for_rotation += t_net_for_rotated * rotated_test_image.shape[0]
            total_s_net_for_rotation += s_net_for_rotated * rotated_test_image.shape[0]

            total_count += rotated_test_image.shape[0]
            original_test_image, rotated_test_image, test_label = cifar10_data.test.next_eval_batch(batch_size)
        
        print("Student Network Accuracy on Original Image: %.4f" % (float(total_s_net_for_original / total_count)))
        print("Teacher Network Accuracy on Original Image: %.4f" % (float(total_t_net_for_original / total_count)))

        print("Student Network Accuracy on Rotated Image: %.4f" % (float(total_s_net_for_rotation / total_count)))
        print("Teacher Network Accuracy on Rotated Image: %.4f" % (float(total_t_net_for_rotation / total_count)))


        print("Start Training...")
        original_train_image, rotated_train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
        end_time_1 = time.time() - start_time
        step = 1
        loss_total = 0
        original_start = start

        while(start != 0):
            #loss_value, train_acc = train_fn(original_train_image, rotated_train_image, train_label)
            ori_mid, rot_mid, train_acc = train_fn(original_train_image, rotated_train_image, train_label)
            step += 1
            # if start == original_start:
            #    print(ori_mid[0])
            #    print(rot_mid[0])
            #    print(train_label)
            
            original_train_image, rotated_train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
            # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            # loss_total += loss_value
        if 1:
            if epoch % 100 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir, 'model_step%d.npy' % epoch)
                weightsOfParams = lasagne.layers.get_all_param_values(rotated_cnn_model)
                np.save(checkpoint_path, weightsOfParams)
                latest_model_path = os.path.join(train_dir, 'latest_model.txt')
                try:
                    os.remove(latest_model_path)
                except OSError:
                    pass
                latest_model_file = open(latest_model_path, "w")
                latest_model_file.write(checkpoint_path)
                latest_model_file.close()

        # print("Epoch Stop, loss_averge", float(loss_total) / float(step))
        duration = time.time() - start_time
        print("Duration is", duration)

        
def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    main()
