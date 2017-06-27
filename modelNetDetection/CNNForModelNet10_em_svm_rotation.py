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
from dataPreparation import load_data
from repeatLayer import Repeat
from affineTransformation3D import AffineTransformation3DLayer
from selectLayer import SelectLayer
from collections import OrderedDict


def NIN_block(input_var, filter_size, num_filters):
    network = Conv2DLayer(input_var, num_filters=num_filters[0],
                          filter_size=(filter_size, filter_size),
                          nonlinearity=lasagne.nonlinearities.identity,
                          W=lasagne.init.GlorotUniform(), pad=(filter_size-1)//2)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)
    network = Conv2DLayer(network, num_filters=num_filters[1],
                          filter_size=(1, 1),
                          nonlinearity=lasagne.nonlinearities.identity,
                          W=lasagne.init.GlorotUniform())
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)
    network = Conv2DLayer(network, num_filters=num_filters[2],
                          filter_size=(1, 1),
                          nonlinearity=lasagne.nonlinearities.identity,
                          W=lasagne.init.GlorotUniform())
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)
    return network


def build_cnn(input_var=None, batch_size=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 32, 32, 32),
                                        input_var=input_var)

    repeatInput = Repeat(network, 10)

    network = lasagne.layers.ReshapeLayer(repeatInput, (-1, 1, 32, 32, 32))

    network_transformed = AffineTransformation3DLayer(network, batch_size * 10)

    network = lasagne.layers.ReshapeLayer(network_transformed, (-1, 32, 32, 32))

    network = Conv2DLayer(
            network, num_filters=5, filter_size=(1, 1),
            # nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.BatchNormLayer(network)

    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)

    network = Conv2DLayer(
            network, num_filters=5, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.BatchNormLayer(network)

    network = Conv2DLayer(
            network, num_filters=1, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)

    network = NIN_block(network, 5, (128, 96, 96))
    network = MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))
    network = lasagne.layers.dropout(network, 0.5)
    network = NIN_block(network, 5, (128, 96, 96))
    network = MaxPool2DLayer(network, pool_size=(2, 2), stride=(2, 2))
    network = lasagne.layers.dropout(network, 0.5)
    network = NIN_block(network, 3, (128, 128, 10))
    network = MaxPool2DLayer(network, pool_size=(8, 8), stride=(1, 1))

    fc2 = lasagne.layers.DenseLayer(
              network,
              num_units=10,
              nonlinearity=lasagne.nonlinearities.identity)

    fc2_selected = SelectLayer(fc2, 10)

    weight_decay_layers = {network: 0.0, fc2: 0.002}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)
    return fc2, fc2_selected, l2_penalty, network_transformed


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
        yield inputs[excerpt], targets[excerpt],\
            indices[start_idx:start_idx+batchsize]


def extend_image(inputs, size=40):
    extended_images = np.zeros((inputs.shape[0], 1, size, size),
                               dtype=np.float32)
    margin_size = (40 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2],
                    margin_size:margin_size + inputs.shape[3]] = inputs
    return extended_images


def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class)
    print("Using all the training data")

    # Load Data##
    X_train, y_train, X_test, y_test = load_data("/X_train.npy",
                                                 "/Y_train.npy",
                                                 "/X_test.npy",
                                                 "/Y_test.npy",
                                                 "MODELNET10")
    X_test_all = X_test
    y_test_all = y_test
    index = np.arange(len(y_test_all))
    np.random.shuffle(index)
    X_test = X_test_all[index[:200]]
    y_test = y_test[index[:200]]

    # Define Batch Size ##
    batch_size = 100

    # Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)

    network, network_for_rotation, weight_decay, network_transformed = \
        build_cnn(input_var, batch_size)

    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    saved_weights = \
        np.load("../data/NIN_ModelNet10_hinge_epoch200.npy")

    affine_matrix_matrix = [np.zeros((batch_size * 10,), dtype=np.float32),
                            np.zeros((batch_size * 10,), dtype=np.float32),
                            np.zeros((batch_size * 10,), dtype=np.float32)]

    network_saved_weights = affine_matrix_matrix + \
        [saved_weights[i] for i in range(saved_weights.shape[0])]

    lasagne.layers.set_all_param_values(network, network_saved_weights)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)

    predictions = lasagne.layers.get_output(network)

    predictions = T.reshape(predictions, (-1, 10, 10))

    predictions_rotation = lasagne.layers.get_output(network_for_rotation,
                                                     deterministic=True)

    predictions_rotation_for_loss = \
        lasagne.layers.get_output(network_for_rotation)

    transformed_images = lasagne.layers.get_output(network_transformed,
                                                   deterministic=True)

    # loss_affine_before = \
    #    lasagne.objectives.squared_error(predictions_rotation.clip(-20, 3), 3) + lasagne.objectives.squared_error(predictions_rotation.clip(3, 20), 20)
    loss_affine_before = lasagne.objectives.squared_error(predictions_rotation.clip(-20, 20), 20)

    # loss_affine_before = -predictions_rotation_for_loss

    loss_affine = loss_affine_before.mean()

    loss = lasagne.objectives.multiclass_hinge_loss(
        predictions_rotation_for_loss, vanilla_target_var)

    loss = loss.mean() + weight_decay

    # This is to use the rotation that the "gradient descent" think will
    # give the highest score for each of the classes
    train_acc_1 = T.mean(T.eq(T.argmax(predictions_rotation_for_loss, axis=1),
                              vanilla_target_var), dtype=theano.config.floatX)

    # This is to use all the scores produces by all the rotations,
    # compare them and get the highest one for each digit
    train_acc_2 = T.mean(T.eq(T.argmax(T.max(predictions, axis=1), axis=1),
                              vanilla_target_var), dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)

    affine_params = params[:3]
    model_params = params[3:]

    updates_affine = lasagne.updates.sgd(loss_affine, affine_params,
                                         learning_rate=400)
    """
    d_loss_wrt_params = T.grad(loss_affine, affine_params)

    updates_affine = OrderedDict()

    for param, grad in zip(affine_params, d_loss_wrt_params):
        updates_affine[param] = param - 1000 * grad
    """
    updates_model = lasagne.updates.adagrad(loss, model_params,
                                            learning_rate=0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1, 10, 10))

    test_prediction_rotation = lasagne.layers.get_output(network_for_rotation,
                                                         deterministic=True)

    test_acc = T.mean(T.eq(T.argmax(T.max(test_prediction, axis=1), axis=1),
                      vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var],
                                     [loss, train_acc_1, train_acc_2],
                                     updates=updates_model)

    train_affine_fn = theano.function([input_var],
                                      [loss_affine, loss_affine_before],
                                      updates=updates_affine)

    val_fn = theano.function([input_var, vanilla_target_var], [test_acc, transformed_images])

    Parallel_Search = True
    TRAIN = True
    TEST = True
    # Finally, launch the training loop.
    # We iterate over epochs:
    cached_affine_matrix = np.array(np.zeros((3, X_train.shape[0], 10)),
                                    dtype=np.float32)

    for epoch in range(num_epochs):
        start_time = time.time()
        if (epoch % 20 == 0 or epoch + 1 == num_epochs) and TEST:
            print ("Start Evaluating...")
            test_acc = 0
            test_batches = 0
            affine_test_batches = 0
            # Find best rotation
            if epoch + 1 == num_epochs and num_epochs != 1:
                X_test = X_test_all
                y_test = y_test_all
            cached_affine_matrix_test = \
                np.array(np.zeros((3, X_test.shape[0], 10)), dtype=np.float32)

            for batch in iterate_minibatches(X_test, y_test,
                                             batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape(batch_size, 32, 32, 32)
                train_loss_before_all = []
                affine_params_all = []

                if Parallel_Search:
                    searchCandidate = 27
                    eachSearchCandidate = 3
                    eachDegree = 90.0 / eachSearchCandidate
                    for x in range(eachSearchCandidate):
                        x_degree = \
                            np.ones((batch_size * 10,), dtype=np.float32) * eachDegree * x - 45.0
                        for y in range(eachSearchCandidate):
                            y_degree = \
                                np.ones((batch_size * 10,), dtype=np.float32) * eachDegree * y - 45.0
                            for z in range(eachSearchCandidate):
                                z_degree = \
                                    np.ones((batch_size * 10,), dtype=np.float32) * eachDegree * z - 45.0

                                affine_params[0].set_value(x_degree)
                                affine_params[1].set_value(y_degree)
                                affine_params[2].set_value(z_degree)

                                for i in range(10):
                                    weightsOfParams = \
                                        lasagne.layers.\
                                        get_all_param_values(network)
                                    train_loss, train_loss_before = \
                                        train_affine_fn(inputs)
                                    """
                                    print("degree x:", weightsOfParams[0][0])
                                    print("degree y:", weightsOfParams[1][0])
                                    print("degree z:", weightsOfParams[2][0])
                                    print("train_loss: ", train_loss_before[0])
                                    print("total_loss: ", train_loss)
                                    print("--------------------------")
                                    """
                                affine_params_all.append(
                                    np.array(weightsOfParams[:3]).reshape(
                                        1, 3, batch_size, 10))

                                train_loss_before_all.append(
                                    train_loss_before.reshape(
                                        1, batch_size, 10))

                else:
                    searchCandidate = 200
                    affine_params[0].set_value(np.zeros((batch_size * 10,),
                                                        dtype=np.float32))
                    affine_params[1].set_value(np.zeros((batch_size * 10,),
                                                        dtype=np.float32))
                    affine_params[2].set_value(np.zeros((batch_size * 10,),
                                                        dtype=np.float32))

                    for i in range(searchCandidate):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_loss, train_loss_before = train_affine_fn(inputs)
                        affine_params_all.append(np.array(weightsOfParams[:3].reshape(1, 3, batch_size, 10)))
                        train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))

                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)

                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape_x = affine_params_all[:, 0].reshape(searchCandidate, -1)
                affine_params_all_reshape_y = affine_params_all[:, 1].reshape(searchCandidate, -1)
                affine_params_all_reshape_z = affine_params_all[:, 2].reshape(searchCandidate, -1)

                # Find the search candidate that gives the lowest loss
                train_arg_min = np.argmin(train_loss_before_all_reshape, axis=0)
                # According to the best search candidate, get the rotations.
                affine_params_all_reshape_x = affine_params_all_reshape_x[train_arg_min, np.arange(train_arg_min.shape[0])]
                affine_params_all_reshape_y = affine_params_all_reshape_y[train_arg_min, np.arange(train_arg_min.shape[0])]
                affine_params_all_reshape_z = affine_params_all_reshape_z[train_arg_min, np.arange(train_arg_min.shape[0])]
                cached_affine_matrix_test[0, index] = affine_params_all_reshape_x.reshape(-1, 10)
                cached_affine_matrix_test[1, index] = affine_params_all_reshape_y.reshape(-1, 10)
                cached_affine_matrix_test[2, index] = affine_params_all_reshape_z.reshape(-1, 10)

                affine_test_batches += 1
                # print(affine_test_batches)

            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets, index = batch
                affine_params[0].set_value(cached_affine_matrix_test[0, index].reshape(-1,))
                affine_params[1].set_value(cached_affine_matrix_test[1, index].reshape(-1,))
                affine_params[2].set_value(cached_affine_matrix_test[2, index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 32, 32, 32)
                acc, transformed_image_result = val_fn(inputs, targets)
                test_acc += acc
                test_batches += 1
                if test_batches == 1:
                    np.save("/hdd/Documents/tmp/3d_original_image_%d.npy" % epoch, inputs)
                    np.save("/hdd/Documents/tmp/3d_transformed_image_%d.npy" % epoch, transformed_image_result)
            print("Final results:")
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

        if not TRAIN:
            break

        print("Starting training...")
        train_err = 0
        train_acc_sum_1 = 0
        train_acc_sum_2 = 0
        train_batches = 0
        affine_train_batches = 0

        if epoch % 20 == 0:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            batch_loss = 0
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape(batch_size, 32, 32, 32)

                train_loss_before_all = []
                affine_params_all = []

                if Parallel_Search:
                    searchCandidate = 27
                    eachSearchCandidate = 3
                    eachDegree = 90.0 / eachSearchCandidate
                    for x in range(eachSearchCandidate):
                        x_degree = np.ones((batch_size * 10,), dtype=np.float32) * eachDegree * x - 45.0
                        for y in range(eachSearchCandidate):
                            y_degree = np.ones((batch_size * 10,), dtype=np.float32) * eachDegree * y - 45.0
                            for z in range(eachSearchCandidate):
                                z_degree = np.ones((batch_size * 10,), dtype=np.float32) * eachDegree * z - 45.0
                                affine_params[0].set_value(x_degree)
                                affine_params[1].set_value(y_degree)
                                affine_params[2].set_value(z_degree)
                                for i in range(10):
                                    weightsOfParams = lasagne.layers.get_all_param_values(network)
                                    train_loss, train_loss_before = train_affine_fn(inputs)
                                affine_params_all.append(np.array(weightsOfParams[:3]).reshape(1, 3, batch_size, 10))
                                train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))

                else:
                    searchCandidate = 200
                    affine_params[0].set_value(np.zeros((batch_size * 10,),
                                                        dtype=np.float32))
                    affine_params[1].set_value(np.zeros((batch_size * 10,),
                                                        dtype=np.float32))
                    affine_params[2].set_value(np.zeros((batch_size * 10,),
                                                        dtype=np.float32))
                    for i in range(searchCandidate):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_loss, train_loss_before = train_affine_fn(inputs)
                        affine_params_all.append(np.array(weightsOfParams[:3].reshape(1, 3, batch_size, 10)))
                        train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))

                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)

                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape_x = affine_params_all[:, 0].reshape(searchCandidate, -1)
                affine_params_all_reshape_y = affine_params_all[:, 1].reshape(searchCandidate, -1)
                affine_params_all_reshape_z = affine_params_all[:, 2].reshape(searchCandidate, -1)

                # Find the search candidate that gives the lowest loss
                train_arg_min = np.argmin(train_loss_before_all_reshape, axis=0)
                # According to the best search candidate, get the rotations.
                affine_params_all_reshape_x = affine_params_all_reshape_x[train_arg_min, np.arange(train_arg_min.shape[0])]
                affine_params_all_reshape_y = affine_params_all_reshape_y[train_arg_min, np.arange(train_arg_min.shape[0])]
                affine_params_all_reshape_z = affine_params_all_reshape_z[train_arg_min, np.arange(train_arg_min.shape[0])]

                cached_affine_matrix[0, index] = affine_params_all_reshape_x.reshape(-1, 10)
                cached_affine_matrix[1, index] = affine_params_all_reshape_y.reshape(-1, 10)
                cached_affine_matrix[2, index] = affine_params_all_reshape_z.reshape(-1, 10)

                affine_train_batches += 1
                batch_loss += np.mean(train_loss_before_all)
                # print(affine_train_batches)
            print(batch_loss / affine_train_batches)
            print (time.time() - start_time)

        if 1:
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets, index = batch
                affine_params[0].set_value(cached_affine_matrix[0, index].reshape(-1,))
                affine_params[1].set_value(cached_affine_matrix[1, index].reshape(-1,))
                affine_params[2].set_value(cached_affine_matrix[2, index].reshape(-1,))

                inputs = inputs.reshape(batch_size, 32, 32, 32)
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

        if epoch % 20 == 19:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            np.save("../data/NIN_ModelNet10_hinge_epoch_%d_3d_find_rotation.npy" % epoch, weightsOfParams)

        # weightsOfParams = lasagne.layers.get_all_param_values(network)
        # np.save("../data/mnist_CNN_params_drop_out_Chi_2017_ROT_hinge_2000_em_new_2.npy", weightsOfParams)


if __name__ == '__main__':
    main()
