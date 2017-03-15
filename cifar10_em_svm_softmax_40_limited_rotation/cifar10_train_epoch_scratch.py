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
train_dir = './cifar10_theano_train_adam'
max_epochs = 2000
validation = False
validation_model = ''

def train():
    """Train CIFAR-10 for a number of steps."""

    input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, cnn_model_for_deformation, weight_decay_penalty, network_transformed = cifar10.build_cnn(input_var, batch_size)

    #saved_weights = np.load(saved_weights_dir)

    #deformation_matrix_matrix = np.array(np.zeros((batch_size * 10, )), dtype = np.float32)

    #network_saved_weights = np.array([deformation_matrix_matrix, ] + [saved_weights[i] for i in range(saved_weights.shape[0])])

    #lasagne.layers.set_all_param_values(cnn_model, network_saved_weights)

    model_output = lasagne.layers.get_output(cnn_model, deterministic=False)#should be false

    model_output = T.reshape(model_output, (-1, 10, 10))

    model_deformation = lasagne.layers.get_output(cnn_model_for_deformation, deterministic = True)

    model_deformation_for_loss = lasagne.layers.get_output(cnn_model_for_deformation, deterministic = False)# should be false

    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)
    
    loss_affine_before = lasagne.objectives.squared_error((model_deformation).clip(-20, 20), 20)

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
        updates_affine[param] = param - 0.0 * grad


    updates_model = lasagne.updates.adam(loss, model_params, learning_rate=0.001)

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

    cifar10_data.train._cached_deformation = np.array(np.zeros((cifar10_data.train._num_examples, 10,)),dtype = np.float32)

    X_test = cifar10_data.sample_test

    for epoch in xrange(max_epochs):
        start_time = time.time() 
        if 1: 
            print("Start Evaluating %d" % epoch)
            affine_test_batches = 0
            # Find the best deformation
            if epoch % 5 == 0 or epoch + 1 == max_epochs:
                if epoch + 1 == max_epochs or epoch % 20 == 5:
                    X_test = cifar10_data.test
                else:
                    X_test = cifar10_data.sample_test
                X_test._cached_deformation = np.array(np.zeros((X_test._num_examples, 10,)), dtype = np.float32)
                test_image, test_label, start = X_test.next_eval_batch(batch_size)

                batch_time = time.time()
                while(test_image is not None):
                    train_loss_before_all = []
                    affine_params_all = []
                    searchCandidate = 8
                    eachDegree = 360.0 / searchCandidate * 0.0
                    for j in range(searchCandidate):
                        affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j, dtype = np.float32))
                        for i in range(1):
                            weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                            train_affine_result = train_affine_fn(test_image)
                            train_loss, train_loss_before, final_transformed_images, _ = train_affine_result[:4]

                        affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                        train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                    train_loss_before_all = np.vstack(train_loss_before_all)
                    affine_params_all = np.vstack(affine_params_all)

                    train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                    affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)
                    
                    # Find the search candidate that gives the lowest loss
                    train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)

                    affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                    X_test._cached_deformation[start:start+batch_size] = affine_params_all_reshape.reshape(-1, 10)
                    affine_test_batches += 1
                    print(affine_test_batches,time.time() - batch_time)
                    batch_time = time.time()
                    test_image, test_label, start = X_test.next_eval_batch(batch_size)

            # Start Evaluation after finding the best deformation
            total_acc_count_1 = 0
            total_acc_count_2 = 0
            total_loss = 0
            total_original_acc_acount = 0
            total_count = 0
            test_image, test_label, start = X_test.next_eval_batch(batch_size)
            while(test_image is not None):
                affine_params.set_value(X_test._cached_deformation[start:start+batch_size].reshape(-1,))
                acc_1, acc_2, test_loss = val_fn(test_image, test_label)
                total_acc_count_1 += acc_1 * test_image.shape[0]
                total_acc_count_2 += acc_2 * test_image.shape[0]
                total_loss += test_loss * test_image.shape[0]
                total_count += test_image.shape[0]
                test_image, test_label, start = X_test.next_eval_batch(batch_size)

            print("Final Results:")
            print("  test accuracy 1:\t\t{:.2f} %".format(
                  float(total_acc_count_1 / total_count) * 100))
            print("  test accuracy 2:\t\t{:.2f} %".format(
                  float(total_acc_count_2 / total_count) * 100))
            print("  test loss:\t\t{:.2f} %".format(
                  float(total_loss / total_count)))
            

        print("Start training...")
        train_err = 0
        train_acc_sum_1 = 0
        train_acc_sum_2 = 0
        train_batches = 0
        loss_total = 0

        start = -1
        if epoch % 5 == 0:
            affine_train_batches = 0
            print("start finding the best affine transformation") 
            batch_loss = 0
            batch_time = time.time()
            while(start + batch_size != cifar10_data.train._num_examples):
                train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 8
                eachDegree = 360.0 / searchCandidate * 0.0
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j, dtype = np.float32))
                    for i in range(1):
                        weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                        train_affine_result = train_affine_fn(train_image)
                        train_loss, train_loss_before, final_transformed_images, model_deformation_result = train_affine_result[:4]
                        #print("gradient of the rotation: ", train_affine_result[4].reshape(-1, 10)[0])
                        #print("Degree of each image", weightsOfParams[0].reshape(-1, 10)[0])
                        #print("Model Deformation Result 0", model_deformation_result.reshape(-1, 10)[0])
                        #print("Quantile", [np.percentile(model_deformation_result, i) for i in range(100)])
                        #print(train_loss_before.reshape(-1, 10)[0])
                        #print(train_loss)
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                    #print("----------------------------------------------------------------------")
                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)

                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)

                train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)

                affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                cifar10_data.train._cached_deformation[start:start+batch_size] = affine_params_all_reshape.reshape(-1, 10)
                
                affine_params.set_value(affine_params_all_reshape)
                train_affine_result = train_affine_fn(train_image)

                affine_train_batches += 1
                
                if affine_train_batches == 1 and epoch % 100 == 0:
                    affine_params.set_value(affine_params_all_reshape)
                    train_affine_result = train_affine_fn(train_image)
                    train_loss, train_loss_before, final_transformed_images, _ = train_affine_result[:4]
                    train_loss_value, train_acc_value_1, train_acc_value_2 = train_fn(train_image, train_label)
                    print(train_loss_value)
                    print(train_acc_value_1)
                    print(train_acc_value_2)
                    print(train_label)
                    #exit()

                batch_loss += np.mean(train_loss_before)
                print(affine_train_batches, start, time.time() - batch_time)
                batch_time = time.time()
        start = -1


        if 1:
            print(np.mean(cifar10_data.train._cached_deformation))
            while(start + batch_size != cifar10_data.train._num_examples):
                train_image, train_label, start = cifar10_data.train.next_batch(batch_size)
                #print(train_label)
                affine_params.set_value(cifar10_data.train._cached_deformation[start:start+batch_size].reshape(-1,))
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
        
            weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
            print(weightsOfParams[0].shape)
            print(weightsOfParams[1].shape)
            
            if epoch % 100 == 0:
                weightsOfParams = lasagne.layers.get_all_param_values(cnn_model)
                np.save("./tmp/CIFAR_CNN_params_ROT_LATENT_LESS_epoch_%d.npy" %epoch, weightsOfParams)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
