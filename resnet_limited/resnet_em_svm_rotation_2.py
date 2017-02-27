#!/usr/bin/env python

"""
Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)

Check the accompanying files for pretrained models. The 32-layer network (n=5), achieves a validation error of 7.42%, 
while the 56-layer network (n=9) achieves error of 6.75%, which is roughly equivalent to the examples in the paper.
"""

from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

from dataPreparation import load_data
from nn_architecture import build_cnn
from collections import OrderedDict

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper : 
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt], indices[start_idx:start_idx + batchsize]

# ############################## Main program ################################

def main(n=5, num_epochs=82, model=None):
    # Check if cifar data exists
    if not os.path.exists("./cifar-10-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = load_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    batch_size = 128

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network, network_selected = build_cnn(input_var, n, batch_size)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
   
    params = None
 
    if model is None:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        prediction_selected = lasagne.layers.get_output(network_selected) 

        with np.load("./cifar10_deep_residual_model.npz") as f:
             saved_weights = [f['arr_%d' % i] for i in range(len(f.files))]

        affine_matrix_matrix = np.array(np.zeros((batch_size * 10,)), dtype = np.float32)
        network_saved_weights = [affine_matrix_matrix,] + [saved_weights[i] for i in range(len(saved_weights))]
 
        lasagne.layers.set_all_param_values(network,network_saved_weights)

        loss_before = lasagne.objectives.multiclass_hinge_loss(prediction_selected, target_var, 10)
        loss = loss_before.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)[4:]
        print(all_layers)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.001
        loss = loss + l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(network, trainable=True)
        affine_params = params[0]
        model_params = params[1:]

        lr = 0.1
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.momentum(
                loss, model_params, learning_rate=sh_lr, momentum=0.9)
        
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)


    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction_selected = lasagne.layers.get_output(network_selected, deterministic=True)

    test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction_selected,
                                                            target_var, 10)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction_selected, axis=1), target_var),
                      dtype=theano.config.floatX)
    
    loss_affine_before = lasagne.objectives.squared_error(prediction_selected.clip(-30, 30), 30)
    loss_affine = loss_affine_before.mean()

    if params is None:
        params = lasagne.layers.get_all_params(network, trainable=True)
        affine_params = params[0]
        model_params = params[1:]

    d_loss_wrt_params = T.grad(loss_affine, [affine_params])
    
    updates_affine = OrderedDict()
    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 0.0 * grad

    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before], updates = 
                                      updates_affine)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    if model is None:
        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        cached_affine_matrix = np.array(np.zeros((X_train.shape[0], 10,)), dtype = np.float32)
        cached_affine_matrix_test = np.array(np.zeros((X_test.shape[0], 10,)), dtype = np.float32)
        for epoch in range(num_epochs):
            # shuffle training data
            train_indices = np.arange(10000)
            #train_indices = np.arange(256)
            np.random.shuffle(train_indices)
            X_train_epoch = X_train[train_indices,:,:,:]
            Y_train_epoch = Y_train[train_indices]
            affine_train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train_epoch, Y_train_epoch, batch_size, shuffle=False, augment=True):
                inputs, targets, indices = batch
                indices = train_indices[indices]
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 6
                eachDegree = 90.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j - 45.0, dtype = np.float32))
                    for i in range(1):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_loss, train_loss_before = train_affine_fn(inputs)
                        #print(weightsOfParams[0].reshape(-1, 10)[0])
                        #print(train_loss)
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)

                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)

                train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)
                affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                cached_affine_matrix[indices] = affine_params_all_reshape.reshape(-1, 10)
                affine_train_batches += 1
                if affine_train_batches % 10 == 0:
                    print(affine_train_batches)

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            for batch in iterate_minibatches(X_train_epoch, Y_train_epoch, batch_size, shuffle=True, augment=True):
                inputs, targets, indices = batch
                indices = train_indices[indices]
                affine_params.set_value(cached_affine_matrix[indices].reshape(-1,))
                train_batch_err = train_fn(inputs, targets)
                train_err += train_batch_err
                train_batches += 1
                """
                if (train_batches == 1):
                    print(train_batch_prediction.reshape(batch_size, 10)[:5])
                    print(loss_batch.reshape(batch_size,)[:5])
                    print(l2_penalty_value)
                """



            for batch in iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
                inputs, targets, indices = batch
                test_loss_before_all = []
                affine_params_all = []
                searchCandidate = 6
                eachDegree = 90.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j - 45.0, dtype = np.float32))
                    for i in range(1):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        test_loss, test_loss_before = train_affine_fn(inputs)
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    test_loss_before_all.append(test_loss_before.reshape(1, batch_size, 10))
                test_loss_before_all = np.vstack(test_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)

                test_loss_before_all_reshape = test_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)

                test_arg_min = np.argmin(test_loss_before_all_reshape, axis = 0)
                affine_params_all_reshape = affine_params_all_reshape[test_arg_min, np.arange(test_arg_min.shape[0])]
                cached_affine_matrix_test[indices] = affine_params_all_reshape.reshape(-1, 10)


            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
                inputs, targets, indices = batch
                affine_params.set_value(cached_affine_matrix_test[indices].reshape(-1,))
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch+1) == 41 or (epoch+1) == 61:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        np.savez('cifar10_deep_residual_model.npz', *lasagne.layers.get_all_param_values(network))
    else:
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # At last, Calculate validation error of model:
    # First, find the best rotation

    for batch in iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
        inputs, targets, indices = batch
        test_loss_before_all = []
        affine_params_all = []
        searchCandidate = 6
        eachDegree = 90.0 / searchCandidate
        for j in range(searchCandidate):
            affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j - 45.0, dtype = np.float32))
            for i in range(1):
                weightsOfParams = lasagne.layers.get_all_param_values(network)
                test_loss, test_loss_before = train_affine_fn(inputs)
            affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
            test_loss_before_all.append(test_loss_before.reshape(1, batch_size, 10))
        test_loss_before_all = np.vstack(test_loss_before_all)
        affine_params_all = np.vstack(affine_params_all)

        test_loss_before_all_reshape = test_loss_before_all.reshape(searchCandidate, -1)
        affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)

        test_arg_min = np.argmin(test_loss_before_all_reshape, axis = 0)
        affine_params_all_reshape = affine_params_all_reshape[test_arg_min, np.arange(test_arg_min.shape[0])]
        cached_affine_matrix_test[indices] = affine_params_all_reshape.reshape(-1, 10)

    # Then, start predicting

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
        inputs, targets, indices = batch
        affine_params.set_value(cached_affine_matrix_test[indices].reshape(-1,))
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
        print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N [MODEL]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['model'] = sys.argv[2]
        main(**kwargs)

