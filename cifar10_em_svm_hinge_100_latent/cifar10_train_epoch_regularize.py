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

import cifar10_regularize as cifar10
import cifar10_input

import theano
import theano.tensor as T
import lasagne

import skimage
import skimage.transform

from collections import OrderedDict

batch_size = 100
saved_weights_dir = '/project/evtimov/jiajun/MultipleDetection/cifar10_em_svm_hinge_100/cifar10_theano_train_hinge_adam_new/model_step25.npy'
train_dir = './cifar10_theano_train_adam_regularize'
max_epochs = 2000
validation = False
validation_model = ''

def train():
    """Train CIFAR-10 for a number of steps."""

    input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, cnn_model_for_deformation, weight_decay_penalty, network_transformed = cifar10.build_cnn(input_var, batch_size)

    saved_weights = np.load(saved_weights_dir)

    deformation_matrix_matrix = np.array(np.zeros((batch_size * 10, 2 * 16)), dtype = np.float32)

    network_saved_weights = np.array([deformation_matrix_matrix, ] + [saved_weights[i] for i in range(saved_weights.shape[0])])

    lasagne.layers.set_all_param_values(cnn_model, network_saved_weights)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=False)

    model_output = T.reshape(model_output, (-1, 10, 10))

    model_deformation = lasagne.layers.get_output(cnn_model_for_deformation, deterministic = True)

    model_deformation_for_loss = lasagne.layers.get_output(cnn_model_for_deformation)

    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)
    
    loss_affine_before = lasagne.objectives.squared_error(model_deformation.clip(-20, 20), 20)

    loss_affine = loss_affine_before.mean() + weight_decay_penalty

    hinge_loss = lasagne.objectives.multiclass_hinge_loss(model_deformation_for_loss, target_var, delta = 20)

    hinge_loss = hinge_loss.mean()

    loss = hinge_loss + weight_decay_penalty

    train_acc_1 = T.mean(T.eq(T.argmax(model_deformation_for_loss, axis = 1), target_var), dtype = theano.config.floatX)

    train_acc_2 = T.mean(T.eq(T.argmax(T.max(model_output, axis = 1), axis = 1), target_var), dtype = theano.config.floatX)


    params = lasagne.layers.get_all_params(cnn_model, trainable=True)

    affine_params = params[0]

    model_params = params[1:]

    d_loss_wrt_params = T.grad(loss_affine, [affine_params])

    updates_affine = OrderedDict()

    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 0.02 * grad


    updates_model = lasagne.updates.adam(loss, model_params, learning_rate=0.001)

    test_prediction = lasagne.layers.get_output(cnn_model, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1, 10, 10))

    test_prediction_deformation = lasagne.layers.get_output(cnn_model_for_deformation, deterministic=True)

    test_acc = T.mean(T.eq(T.argmax(T.max(test_prediction, axis = 1), axis = 1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], [loss, train_acc_1, train_acc_2], updates = updates_model)

    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before, transformed_images], updates=updates_affine)

    val_fn = theano.function([input_var, target_var], test_acc)

    if os.path.isfile(os.path.join(train_dir, 'latest_model.txt')):
        weight_file = ""
        with open(os.path.join(train_dir, 'latest_model.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        lasagne.layers.set_all_param_values(cnn_model, model_weights)



    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_input.load_cifar10()

    cifar10_data.train._cached_deformation = np.array(np.zeros((cifar10_data.train._num_examples, 10, 2 * 16)),dtype = np.float32)

    X_test = cifar10_data.sample_test

    for epoch in xrange(max_epochs):
        
        start_time = time.time() 
        if epoch % 50 == 0 or epoch + 1 == max_epochs:
            print("Start Evaluating %d" % epoch)
            total_acc_count = 0
            total_count = 0
            affine_test_batches = 0
            if epoch + 1 == max_epochs:
                X_test = cifar10_data.test
            X_test._cached_deformation = np.array(np.zeros((X_test._num_examples, 10, 2 * 16)), dtype = np.float32)
            # Find the best deformation
            test_image, test_label, start = X_test.next_eval_batch(batch_size)

            while(test_image is not None):
                affine_params.set_value(np.array(np.zeros((batch_size * 10, 2 * 16)), dtype = np.float32))
                for i in range(200):
                    weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                    train_loss, train_loss_before, final_transformed_images = train_affine_fn(test_image)
                X_test._cached_deformation[start:start+batch_size] = weightsOfParams[0].reshape((-1, 10, 2 * 16))
                affine_test_batches += 1
                print(affine_test_batches)
                test_image, test_label, start = X_test.next_eval_batch(batch_size)

            # Start Evaluation after finding the best deformation
            test_image, test_label, start = X_test.next_eval_batch(batch_size)
            while(test_image is not None):
                affine_params.set_value(X_test._cached_deformation[start:start+batch_size].reshape(-1, 2 * 16))
                acc = val_fn(test_image, test_label)
                total_acc_count += acc * test_image.shape[0]
                total_count += test_image.shape[0]
                test_image, test_label, start = X_test.next_eval_batch(batch_size)

            print("Final Results:")
            print("  test accuracy:\t\t{:.2f} %".format(
                  float(total_acc_count / total_count) * 100))

        print("Start training...")
        train_err = 0
        train_acc_sum_1 = 0
        train_acc_sum_2 = 0
        train_batches = 0
        loss_total = 0

        start = -1
        if epoch % 200 == 0:
            affine_train_batches = 0
            print("start finding the best affine transformation") 
            batch_loss = 0
            while(start!=0 or affine_train_batches==1):
                train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
                affine_params.set_value(np.array(np.zeros((batch_size * 10, 2 * 16)), dtype = np.float32))
                final_transformed_images = None
                for i in range(200):
                    weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                    train_loss, train_loss_before, final_transformed_images = train_affine_fn(train_image)
                cifar10_data.train._cached_deformation[start:start+batch_size] = weightsOfParams[0].reshape((-1, 10, 2 * 16))
                affine_train_batches += 1
                batch_loss += np.mean(train_loss_before)
                print(affine_train_batches)
            
            print(batch_loss / affine_train_batches)
            print(time.time() - start_time)
            cifar10_data.train.reset()

        start = -1

        if 1:
            while(start!=0 or train_batches==1):
                train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
                affine_params.set_value(cifar10_data.train._cached_deformation[start:start+batch_size].reshape(-1, 2 * 16))
                train_loss_value, train_acc_value_1, train_acc_value_2 = train_fn(train_image, train_label)
                train_err += train_loss_value
                train_acc_sum_1 += train_acc_value_1
                train_acc_sum_2 += train_acc_value_2
                train_batches += 1
            cifar10_data.train.reset()

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc 1:\t\t{:.6f}".format(train_acc_sum_1 / train_batches))
        print("  training acc 2:\t\t{:.6f}".format(train_acc_sum_2 / train_batches))

        if epoch % 100 == 0 or (epoch + 1) == max_epochs:
            checkpoint_path = os.path.join(train_dir, 'model_epoch%d.npy' % epoch)
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


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    main()
