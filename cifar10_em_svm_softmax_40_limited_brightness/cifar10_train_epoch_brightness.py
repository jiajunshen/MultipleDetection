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

from collections import OrderedDict

batch_size = 100
#saved_weights_dir = '/project/evtimov/jiajun/MultipleDetection/cifar10_em_svm_hinge_100/cifar10_theano_train_hinge_adam_new/model_step25.npy'
saved_weights_dir = '../cifar10_em_svm_softmax_40_hsv/cifar10_theano_train_adam/model_step20000.npy'
train_dir = './cifar10_theano_train_adam'
max_epochs = 2000
validation = False
validation_model = ''


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], indices[start_idx:start_idx+batchsize]


def train():
    """Train CIFAR-10 for a number of steps."""

    input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, cnn_model_for_deformation, weight_decay_penalty, network_transformed = cifar10.build_cnn(input_var, batch_size)

    saved_weights = np.load(saved_weights_dir)

    deformation_matrix_matrix = np.array(np.zeros((batch_size * 10, )), dtype = np.float32)

    network_saved_weights = np.array([deformation_matrix_matrix, ] + [saved_weights[i] for i in range(saved_weights.shape[0])])

    lasagne.layers.set_all_param_values(cnn_model, network_saved_weights)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=False)#should be false

    model_output = T.reshape(model_output, (-1, 10, 10))

    model_deformation = lasagne.layers.get_output(cnn_model_for_deformation, deterministic = True)

    model_deformation_for_loss = lasagne.layers.get_output(cnn_model_for_deformation, deterministic = False)# should be false

    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)
    
    loss_affine_before = lasagne.objectives.squared_error((model_deformation).clip(-100, 100), 100)

    loss_affine = loss_affine_before.mean() + weight_decay_penalty

    model_deformation_for_loss_softmax = lasagne.nonlinearities.softmax(model_deformation_for_loss)

    categorical_loss = lasagne.objectives.categorical_crossentropy(model_deformation_for_loss_softmax, target_var)

    categorical_loss = categorical_loss.mean()

    loss = categorical_loss

    train_acc_1 = T.mean(T.eq(T.argmax(model_deformation_for_loss_softmax, axis = 1), target_var), dtype = theano.config.floatX)

    train_acc_2 = T.mean(T.eq(T.argmax(T.max(model_output, axis = 1), axis = 1), target_var), dtype = theano.config.floatX)


    params = lasagne.layers.get_all_params(cnn_model, trainable=True)

    affine_params = params[0]

    model_params = params[1:]

    d_loss_wrt_params = T.grad(loss_affine, [affine_params])

    updates_affine = OrderedDict()

    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 0.004 * grad

    updates_model = lasagne.updates.adam(loss, model_params, learning_rate=0.0001)

    test_prediction = lasagne.layers.get_output(cnn_model, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1, 10, 10))

    test_prediction_deformation = lasagne.layers.get_output(cnn_model_for_deformation, deterministic=True)
    
    test_prediction_deformation_softmax = lasagne.nonlinearities.softmax(test_prediction_deformation)

    test_acc_1 = T.mean(T.eq(T.argmax(T.max(test_prediction, axis = 1), axis = 1), target_var),
                      dtype=theano.config.floatX)
    
    test_acc_2 = T.mean(T.eq(T.argmax(test_prediction_deformation_softmax, axis = 1), target_var),
                      dtype=theano.config.floatX)

    test_loss_before = lasagne.objectives.categorical_crossentropy(test_prediction_deformation_softmax, target_var)

    test_loss = test_loss_before.mean()

    train_fn = theano.function([input_var, target_var], [loss, train_acc_1, train_acc_2], updates = updates_model)

    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before, transformed_images, model_deformation] + d_loss_wrt_params, updates=updates_affine)

    val_fn = theano.function([input_var, target_var], [test_acc_1, test_acc_2, test_loss])

    if os.path.isfile(os.path.join(train_dir, 'latest_model.txt')):
        weight_file = ""
        with open(os.path.join(train_dir, 'latest_model.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        lasagne.layers.set_all_param_values(cnn_model, model_weights)

    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_input.load_cifar10()

    X_train = cifar10_data.train._images
    Y_train = cifar10_data.train._labels

    print("Training Size and range", X_train.shape, np.max(X_train), np.min(X_train))

    X_test = cifar10_data.test._images
    Y_test = cifar10_data.test._labels

    print("Testing Size and range", X_test.shape, np.max(X_test), np.min(X_test))

    X_sample_test = cifar10_data.sample_test._images
    Y_sample_test = cifar10_data.sample_test._labels

    print("Sample Testing Size and range", X_sample_test.shape, np.max(X_sample_test), np.min(X_sample_test))



    training_cached_deformation = np.array(np.zeros((X_train.shape[0], 10,)),dtype = np.float32)

    for epoch in xrange(max_epochs):
        start_time = time.time() 
        if 1: 
            print("Start Evaluating %d" % epoch)
            affine_test_batches = 0
            # Find the best deformation
            if epoch % 20 == 0 or epoch + 1 == max_epochs:
                if (epoch + 1 == max_epochs or epoch % 100 == 0) and epoch != 0:
                    X_current_test = X_test
                    Y_current_test = Y_test
                else:
                    X_current_test = X_sample_test
                    Y_current_test = Y_sample_test
                testing_cached_deformation = np.array(np.zeros((X_current_test.shape[0], 10)), dtype = np.float32)

                for batch in iterate_minibatches(X_current_test, Y_current_test, batch_size):
                    test_image, test_label, test_indices = batch 
                    batch_time = time.time()
                    train_loss_before_all = []
                    affine_params_all = []
                    searchCandidate = 8
                    eachDegree = 1.0 / searchCandidate
                    for j in range(searchCandidate):
                        affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j - 0.5, dtype = np.float32))
                        for i in range(20):
                            weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                            train_affine_result = train_affine_fn(test_image)
                            train_loss, train_loss_before, final_transformed_images, model_deformation_result = train_affine_result[:4]
                            """
                            if epoch % 5 == 0:
                                print(searchCandidate, i)
                                print("gradient of the rotation: ", train_affine_result[4].reshape(-1, 10)[0])
                                print("Brightness of each image", weightsOfParams[0].reshape(-1, 10)[0])
                                print("Model Deformation Result 0", model_deformation_result.reshape(-1, 10)[0])
                                print("Quantile", [np.percentile(model_deformation_result, i) for i in range(100)])
                                print(train_loss_before.reshape(-1, 10)[0])
                                print(train_loss)
                                print("----------------------------------------------------------------------")
                            
                            print("==========================================================================")
                            """

                        affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                        train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                    train_loss_before_all = np.vstack(train_loss_before_all)
                    affine_params_all = np.vstack(affine_params_all)

                    train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                    affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)
                    
                    # Find the search candidate that gives the lowest loss
                    train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)

                    affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                    testing_cached_deformation[test_indices] = affine_params_all_reshape.reshape(-1, 10)
                    affine_test_batches += 1
                    print(affine_test_batches,time.time() - batch_time)
                    batch_time = time.time()

            # Start Evaluation after finding the best deformation
            total_acc_count_1 = 0
            total_acc_count_2 = 0
            total_loss = 0
            total_original_acc_acount = 0
            total_count = 0
            for batch in iterate_minibatches(X_current_test, Y_current_test, batch_size):
                test_image, test_label, test_indices = batch
                affine_params.set_value(testing_cached_deformation[test_indices].reshape(-1,))
                acc_1, acc_2, test_loss = val_fn(test_image, test_label)
                total_acc_count_1 += acc_1 * test_image.shape[0]
                total_acc_count_2 += acc_2 * test_image.shape[0]
                total_loss += test_loss * test_image.shape[0]
                total_count += test_image.shape[0]

            print("Final Results:")
            print("  test accuracy 1:\t\t{:.2f} %".format(
                  float(total_acc_count_1 / total_count) * 100))
            print("  test accuracy 2:\t\t{:.2f} %".format(
                  float(total_acc_count_2 / total_count) * 100))
            print("  test loss:\t\t{:.2f}".format(
                  float(total_loss / total_count)))
            

        print("Start training...")
        train_err = 0
        train_acc_sum_1 = 0
        train_acc_sum_2 = 0
        train_batches = 0
        loss_total = 0

        if epoch % 100 == 0:
            print("start finding the best affine transformation") 
            affine_train_batches = 0
            batch_time = time.time()
            for batch in iterate_minibatches(X_train, Y_train, batch_size):
                train_image, train_label, train_indices = batch
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 8
                eachDegree = 1.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j - 0.5, dtype = np.float32))
                    for i in range(20):
                        weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                        train_affine_result = train_affine_fn(train_image)
                        train_loss, train_loss_before, final_transformed_images, model_deformation_result = train_affine_result[:4]
                        """
                        if epoch % 5 == 0 and affine_train_batches == 1:
                            print("gradient of the rotation: ", train_affine_result[4].reshape(-1, 10)[0])
                            print("Brightness of each image", weightsOfParams[0].reshape(-1, 10)[0])
                            print("Model Deformation Result 0", model_deformation_result.reshape(-1, 10)[0])
                            print("Quantile", [np.percentile(model_deformation_result, i) for i in range(100)])
                            print(train_loss_before.reshape(-1, 10)[0])
                            print(train_loss)
                            print("----------------------------------------------------------------------")
                        """
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))

                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)

                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)

                train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)

                affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                training_cached_deformation[train_indices] = affine_params_all_reshape.reshape(-1, 10)
                
                affine_train_batches += 1
                
                if affine_train_batches == 1 and epoch % 100 == 0:
                    affine_params.set_value(affine_params_all_reshape)
                    train_affine_result = train_affine_fn(train_image)
                    train_loss, train_loss_before, final_transformed_images, _ = train_affine_result[:4]
                    train_loss_value, train_acc_value_1, train_acc_value_2 = train_fn(train_image, train_label)
                    print(train_loss_value)
                    print(train_acc_value_1)
                    print(train_acc_value_2)
                    print(np.array([affine_params_all_reshape.reshape(-1, 10)[i, train_label[i]] for i in range(batch_size)]))
                    np.save("./transformed_image.npy", final_transformed_images)

                print(affine_train_batches, time.time() - batch_time)
                batch_time = time.time()


        if 1:
            for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle = True):
                train_image, train_label, train_indices = batch
                affine_params.set_value(training_cached_deformation[train_indices].reshape(-1,))
                train_loss_value, train_acc_value_1, train_acc_value_2 = train_fn(train_image, train_label)
                train_err += train_loss_value
                train_acc_sum_1 += train_acc_value_1
                train_acc_sum_2 += train_acc_value_2
                train_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, max_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  training acc 1:\t\t{:.6f}".format(train_acc_sum_1 / train_batches))
            print("  training acc 2:\t\t{:.6f}".format(train_acc_sum_2 / train_batches))
        
            if epoch % 100 == 0:
                weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                np.save("./tmp/CIFAR_CNN_params_ROT_LATENT_LESS_epoch_%d.npy" %epoch, weightsOfParams)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
