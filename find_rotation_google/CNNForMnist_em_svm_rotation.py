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
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data
from repeatLayer import Repeat
from rotationMatrixLayer import RotationTransformationLayer
from selectLayer import SelectLayer
from collections import OrderedDict

def binary_hinge_loss(predictions, targets, delta=1, log_odds=True,
                      binary=True):
    if log_odds is None:  # pragma: no cover
        raise FutureWarning(
                "The `log_odds` argument to `binary_hinge_loss` will change "
                "its default to `False` in a future version. Explicitly give "
                "`log_odds=True` to retain current behavior in your code, "
                "but also check the documentation if this is what you want.")
        log_odds = True
    if not log_odds:
        predictions = theano.tensor.log(predictions / (1 - predictions))
    if binary:
        targets = 2 * targets - 1
    predictions, targets = align_targets(predictions, targets)
    print(predictions.shape)
    print(targets.shape)
    return theano.tensor.nnet.relu(delta - predictions * targets)

def align_targets(predictions, targets):
    if (getattr(predictions, 'broadcastable', None) == (False, True) and
            getattr(targets, 'ndim', None) == 1):
        targets = as_theano_expression(targets).dimshuffle(0, 'x')
    return predictions, targets

def build_cnn(input_var=None, batch_size=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 68, 68),
                                        input_var=input_var)

    reshapeInput = lasagne.layers.ReshapeLayer(network, (-1, 3, 68, 68))

    original_transformed = RotationTransformationLayer(reshapeInput, batch_size)

    input_transformed = lasagne.layers.SliceLayer(original_transformed, indices=slice(10, 58), axis = 2)

    input_transformed = lasagne.layers.SliceLayer(input_transformed, indices=slice(10, 58), axis = 3)

    norm0 = BatchNormLayer(network)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = Conv2DLayer(
            norm0, num_filters=16, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    network = BatchNormLayer(network)

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = Conv2DLayer(
            network, num_filters=16, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = BatchNormLayer(network)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=128,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=1,
            nonlinearity = lasagne.nonlinearities.identity
            )

    return network, input_transformed



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

def extend_image(images, dim = 68):
    extended_images_res = np.pad(images, ((0,), (0,), (10,),(10,)), mode="reflect")
    return extended_images_res 


def main(model='mlp', num_epochs=2000):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class) 
    print("Using all the training data") 
    
    ## Load Data##
    X_train, y_train, X_test, y_test = load_data("/X_train_rotated.npy", "/Y_train_rotated.npy", "/X_test_rotated.npy", "/Y_test_rotated.npy")


    ## Define Batch Size ##
    batch_size = 100
 
    ## Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)
    
    network, network_transformed = build_cnn(input_var, batch_size)
    
    saved_weights = np.load("../data/google_car_CNN_params_drop_out_Chi_2017_hinge.npy")

    affine_matrix_matrix = np.array(np.zeros((batch_size,)), dtype = np.float32)
    
    network_saved_weights = np.array([affine_matrix_matrix,] + [saved_weights[i] for i in range(saved_weights.shape[0])])
    
    lasagne.layers.set_all_param_values(network, network_saved_weights)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)

    predictions = lasagne.layers.get_output(network)

    predictions = T.reshape(predictions, (-1,))
    
    predictions_rotation = lasagne.layers.get_output(network, deterministic = True)
    
    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)

    
    
    #loss_affine_before = lasagne.objectives.squared_error(predictions_rotation.clip(-20, 3), 3) + lasagne.objectives.squared_error(predictions_rotation.clip(3, 20), 20)
    loss_affine_before = lasagne.objectives.squared_error(predictions_rotation.clip(-20, 20), 20)

    loss_affine = loss_affine_before.mean()

    loss = binary_hinge_loss(predictions, vanilla_target_var)
    loss = loss.mean() + weight_decay

    # This is to use all the scores produces by all the rotations, compare them and get the highest one for each digit
    train_acc = T.mean(T.eq((predictions > 0), vanilla_target_var), dtype = theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    affine_params = params[0]
    model_params = params[1:]
    
    # updates_affine = lasagne.updates.sgd(loss_affine, [affine_params], learning_rate = 10)
    d_loss_wrt_params = T.grad(loss_affine, [affine_params])
    
    updates_affine = OrderedDict()

    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 1000 * grad

    
    #updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    updates_model = lasagne.updates.sgd(loss, params, learning_rate=0.00001)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1,))
    
    test_acc = T.mean(T.eq((test_prediction > 0), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var], [loss,train_acc], updates=updates_model)
    
    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before], updates=updates_affine)

    val_fn = theano.function([input_var, vanilla_target_var], test_acc)

    # Finally, launch the training loop.
    # We iterate over epochs:
    cached_affine_matrix = np.array(np.zeros((X_train.shape[0],)), dtype = np.float32)
    for epoch in range(num_epochs):
        start_time = time.time()
        print ("Start Evaluating...")
        test_acc = 0
        test_batches = 0
        affine_test_batches = 0
        # Find best rotation
        cached_affine_matrix_test = np.array(np.zeros((X_test.shape[0],)), dtype = np.float32)
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
            inputs, targets, index = batch
            inputs = inputs.reshape(batch_size, 3, 68, 68)
            train_loss_before_all = []
            affine_params_all = []
            searchCandidate = 6
            eachDegree = 360.0 / searchCandidate
            for j in range(searchCandidate):
                affine_params.set_value(np.array(np.ones(batch_size) * eachDegree * j, dtype = np.float32))
                for i in range(10):
                    weightsOfParams = lasagne.layers.get_all_param_values(network)
                    train_loss, train_loss_before = train_affine_fn(inputs)
            
                affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 1)))
                train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 1))
            train_loss_before_all = np.vstack(train_loss_before_all)
            affine_params_all = np.vstack(affine_params_all)
            
            train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
            affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)
            
            # Find the search candidate that gives the lowest loss
            train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)
            # According to the best search candidate, get the rotations.
            affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
            cached_affine_matrix_test[index] = affine_params_all_reshape.reshape(-1,)
            affine_test_batches += 1
            print(affine_test_batches)

            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
                inputs, targets, index = batch
                affine_params.set_value(cached_affine_matrix_test[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 3, 68, 68)
                acc = val_fn(inputs, targets)
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
        print("Starting training...")

        train_err = 0
        train_acc_sum_1 = 0
        train_batches = 0
        affine_train_batches = 0

        weightsOfParams = lasagne.layers.get_all_param_values(network)
        batch_loss = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets, index = batch
            inputs = inputs.reshape(batch_size, 3, 68, 68)
            
            train_loss_before_all = []
            affine_params_all = []
            searchCandidate = 6
            eachDegree = 360.0 / searchCandidate
            for j in range(searchCandidate):
                affine_params.set_value(np.array(np.ones(batch_size) * eachDegree * j, dtype = np.float32))
                for i in range(10):
                    weightsOfParams = lasagne.layers.get_all_param_values(network)
                    train_loss, train_loss_before = train_affine_fn(inputs)
                
                affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 1)))
                train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 1))
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
            
            cached_affine_matrix[index] = affine_params_all_reshape.reshape(-1,)
            affine_train_batches += 1
            batch_loss += np.mean(train_loss_before_all)
            print(affine_train_batches) 
        print(batch_loss / affine_train_batches)
        print (time.time() - start_time)

        if 1:
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets, index = batch
                affine_params.set_value(cached_affine_matrix[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 3, 68, 68)
                train_loss_value, train_acc_value_1 = train_model_fn(inputs, targets)
                train_err += train_loss_value
                train_acc_sum_1 += train_acc_value_1
                train_batches += 1
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc 1:\t\t{:.6f}".format(train_acc_sum_1 / train_batches))
        
        if epoch % 100 == 0:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            np.save("../data/google_earth_em_new_batch_run_epoch_%d.npy" %epoch, weightsOfParams)

        
if __name__ == '__main__':
    main()
