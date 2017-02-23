# This file is a editted version of https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
# Use Lasagne for digit recognition using  MNIST dataset.
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
import cv2
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.regularization import regularize_layer_params_weighted, l2
from  lasagne.layers import LocalResponseNormalization2DLayer as LRN
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data
from repeatLayer import Repeat
from rotationMatrixLayer import RotationTransformationLayer
from selectLayer import SelectLayer
from collections import OrderedDict

def build_cnn(input_var=None, batch_size = None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 227, 227),
                                        input_var=input_var)

    repeatInput = Repeat(network, 10)

    network = lasagne.layers.ReshapeLayer(repeatInput, (-1, 1, 227, 227))
    
    network_transformed = RotationTransformationLayer(network, batch_size * 10)

    network = Conv2DLayer(
            network_transformed, num_filters=96, filter_size=(11, 11),
            stride=(4,4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = LRN(network, alpha = 0.0001, beta = 0.75, n = 5)

    # Max-pooling layer of factor 2 in both dimensions:
    network = MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = Conv2DLayer(
            network, num_filters=256, filter_size=(5, 5),
            pad = 2,
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    
    network = LRN(network, alpha = 0.0001, beta = 0.75, n = 5)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride = (2, 2))
    

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1  = lasagne.layers.DenseLayer(
            network,
            num_units=256,
            W = lasagne.init.Normal(0.01),
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            fc1,
            num_units=4096,
            W = lasagne.init.Normal(0.005),
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    fc3  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc2, p=.5),
            num_units=4096,
            W = lasagne.init.Normal(0.005),
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    fc4  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc3, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.identity,
            )

    network_transformed = lasagne.layers.ReshapeLayer(network_transformed, (-1, 10, 40, 40))

    fc4_selected = SelectLayer(fc4, 10)
    
    weight_decay_layers = {fc1:0.0, fc2:0.0}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc4, fc4_selected, l2_penalty, network_transformed


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


def extend_images(images, dim = 277):
    extended_images_res = np.pad(images, ((0,0),(0,0),(13,14),(13,14)), mode="wrap")
    return extended_images_res


def main(model='mlp', num_epochs=2000):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class) 
    print("Using all the training data") 
    
    ## Load Data##
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

    X_train = X_train[y_train < 10]
    X_test = X_test[y_test < 10]

    y_train = y_train[y_train < 10]
    y_test = y_test[y_test < 10]
    
    X_train = extend_images(X_train, 227)
    X_test = extend_images(X_test, 227)

    y_train = y_train
    y_test = y_test


    ## Define Batch Size ##
    batch_size = 20
 
    ## Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)
    
    network, network_for_rotation, weight_decay, network_transformed = build_cnn(input_var, batch_size)
    
    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    saved_weights = np.load("/var/tmp/result.npy")

    affine_matrix_matrix = np.array(np.zeros((batch_size * 10,)), dtype = np.float32)
    
    network_saved_weights = np.array([affine_matrix_matrix,] + [saved_weights[i] for i in range(saved_weights.shape[0])])
    
    lasagne.layers.set_all_param_values(network, network_saved_weights)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)

    predictions = lasagne.layers.get_output(network)

    predictions = T.reshape(predictions, (-1, 10, 10))
    
    predictions_rotation = lasagne.layers.get_output(network_for_rotation, deterministic = True)
    
    predictions_rotation_for_loss = lasagne.layers.get_output(network_for_rotation)
    
    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)

    
    #loss_affine_before = lasagne.objectives.squared_error(predictions_rotation.clip(-20, 20), 20)
    loss_affine_before = lasagne.objectives.squared_error(predictions_rotation.clip(-10, 10), 10)

    loss_affine = loss_affine_before.mean()

    loss = lasagne.objectives.multiclass_hinge_loss(predictions_rotation_for_loss, vanilla_target_var, 5)
    loss = loss.mean() + weight_decay

    # This is to use the rotation that the "gradient descent" think will give the highest score for each of the classes
    train_acc_1 = T.mean(T.eq(T.argmax(predictions_rotation_for_loss, axis = 1), vanilla_target_var), dtype = theano.config.floatX)
    
    # This is to use all the scores produces by all the rotations, compare them and get the highest one for each digit
    train_acc_2 = T.mean(T.eq(T.argmax(T.max(predictions, axis = 1), axis = 1), vanilla_target_var), dtype = theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    

    affine_params = params[0]
    model_params = params[1:]
    
    # updates_affine = lasagne.updates.sgd(loss_affine, [affine_params], learning_rate = 10)
    d_loss_wrt_params = T.grad(loss_affine, [affine_params])
    
    updates_affine = OrderedDict()

    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 1000 * grad

    
    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1, 10, 10))
    
    test_acc = T.mean(T.eq(T.argmax(T.max(test_prediction, axis=1), axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var], [loss,train_acc_1, train_acc_2], updates=updates_model)
    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before,predictions_rotation] + d_loss_wrt_params, updates=updates_affine)

    val_fn = theano.function([input_var, vanilla_target_var], test_acc)
    

    # Finally, launch the training loop.
    # We iterate over epochs:
    cached_affine_matrix = np.array(np.zeros((X_train.shape[0], 10,)), dtype = np.float32)
    for epoch in range(num_epochs):
        start_time = time.time()
        if epoch % 10 == 0 or epoch + 1 == num_epochs:
            print ("Start Evaluating...")
            test_acc = 0
            test_batches = 0
            affine_test_batches = 0
            # Find best rotation
            cached_affine_matrix_test = np.array(np.zeros((X_test.shape[0], 10,)), dtype = np.float32)


            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape(batch_size, 1, 227, 227)
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 8
                eachDegree = 360.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j, dtype = np.float32))
                    for i in range(10):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_affine_res = train_affine_fn(inputs)
                        train_loss, train_loss_before = train_affine_res[:2]
                
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)
                
                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)
                
                # Find the search candidate that gives the lowest loss
                train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)
                # According to the best search candidate, get the rotations.
                affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                cached_affine_matrix_test[index] = affine_params_all_reshape.reshape(-1, 10)
                affine_test_batches += 1
                print(affine_test_batches)

            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
                inputs, targets, index = batch
                affine_params.set_value(cached_affine_matrix_test[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 1, 227, 227)
                acc = val_fn(inputs, targets)
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
        print("Starting training...")
        train_err = 0
        train_acc_sum_1 = 0
        train_acc_sum_2 = 0
        train_batches = 0
        affine_train_batches = 0

        if epoch % 10 == 0:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            batch_loss = 0
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape(batch_size, 1, 227, 227)
                
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 8
                eachDegree = 360.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j, dtype = np.float32))
                    for i in range(10):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_affine_res = train_affine_fn(inputs)
                        train_loss, train_loss_before = train_affine_res[:2]
                        #print(train_loss_before.reshape(-1, 61)[0])
                        #print(np.array(train_affine_res[2]).reshape(batch_size, 61)[0])
                        #print([np.percentile(train_affine_res[2], i) for i in range(100)])
                        #print(np.array(train_affine_res[3:]).reshape(batch_size, 61)[0])
                        #print(np.array(weightsOfParams[0]).reshape(batch_size, 61)[0])
                        #print(train_loss)
                    
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                    #print("---------------------------------------------------------------------")
                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)
                
                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)
                
                # Find the search candidate that gives the lowest loss
                train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)

                # According to the best search candidate, get the rotations.
                affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                
                train_loss_before_all = np.min(train_loss_before_all, axis = 0)
                
                cached_affine_matrix[index] = affine_params_all_reshape.reshape(-1, 10)
                affine_train_batches += 1
                batch_loss += np.mean(train_loss_before_all)
                print(affine_train_batches) 
            print(batch_loss / affine_train_batches)
            print (time.time() - start_time)

        if 1:
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets, index = batch
                affine_params.set_value(cached_affine_matrix[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 1, 227, 227)
                train_loss_value, train_acc_value_1, train_acc_value_2 = train_model_fn(inputs, targets)
                train_err += train_loss_value
                train_acc_sum_1 += train_acc_value_1
                train_acc_sum_2 += train_acc_value_2
                train_batches += 1
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc 1:\t\t{:.6f}".format(train_acc_sum_1 / train_batches))
        print("  training acc 2:\t\t{:.6f}".format(train_acc_sum_2 / train_batches))
        
        if epoch % 20 == 0:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            np.save("/var/tmp/Curet_CNN_Hinge_epoch_%d_2pool_8rotation_400_epoch_10_classes.npy" %epoch, weightsOfParams)

        


if __name__ == '__main__':
    main()
