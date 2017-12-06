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
from dataPreparation import load_data_digit_clutter
from repeatLayer import Repeat
from TPSTransformationMatrixLayer import TPSTransformationMatrixLayer
from selectLayer import SelectLayer
from CNNForMnist_Rotation_Net import build_cnn as build_rotation_cnn
from CNNForMnist_Rotation_Net import rotateImage_batch
from CNNForMnist_Rotation_Net import rotateImage
from collections import OrderedDict

def build_cnn(input_var=None, support_var = None, batch_size = None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(batch_size, 1, 40, 40),
                                        input_var=input_var)
    
    support_input = lasagne.layers.InputLayer(shape=(batch_size, 10, 40, 40),
                                              input_var=support_var)

    repeatInput = Repeat(network, 10)

    network = lasagne.layers.ReshapeLayer(repeatInput, (-1, 1, 40, 40))

    network_transformed_TPS = TPSTransformationMatrixLayer(network, batch_size * 10)
    
    network_transformed_TPS_reshape = lasagne.layers.ReshapeLayer(network_transformed_TPS, (-1, 10, 40, 40))

    after_support_layer = lasagne.layers.ElemwiseMergeLayer([network_transformed_TPS_reshape, support_input], T.mul)

    after_support_layer = lasagne.layers.ReshapeLayer(after_support_layer, (-1 , 1, 40, 40))

    network = Conv2DLayer(
            after_support_layer, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            )
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc1, p=.5),
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=10,
            )

    network_transformed = lasagne.layers.ReshapeLayer(after_support_layer, (-1, 10, 40, 40))

    fc2_selected = SelectLayer(fc2, 10)

    weight_decay_layers = {network_transformed_TPS:0.1}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc2, fc2_selected, l2_penalty, network_transformed, network

def build_linear(input_var=None, batch_size = None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(batch_size, 32, 7, 7),
                                        input_var=input_var)

    network = lasagne.layers.ReshapeLayer(network, (-1, 320, 7, 7))
    
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc1, p=.5),
            nonlinearity=lasagne.nonlinearities.softmax,
            num_units=10,
            )

    return fc2

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


def extend_image(inputs, size = 40):
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    margin_size = (40 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images


def main(model='mlp', num_epochs=1000):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class)
    print("Using all the training data")

    ## Load Data##
    X_train, y_train, X_test, y_test = load_data("/X_train_limited_100.npy", "/Y_train_limited_100.npy", "/X_test.npy", "/Y_test.npy")
    #_, _, X_test, y_test = load_data("/mnistMoreClutter.npy", "/mnistMoreClutterLabel.npy", "/mnistMoreClutterTest.npy", "/mnistMoreClutterLabelTest.npy", W=40)
    _, _, X_test, y_test = load_data_digit_clutter("/X_train_limited_100.npy", "/Y_train_limited_100.npy", "/X_test.npy", "/Y_test.npy")

    X_train = extend_image(X_train, 40)
    X_test_all = extend_image(X_test, 40)
    X_test = extend_image(X_test, 40)

    y_train = y_train
    y_test_all = y_test[:]

    """
    X_train = X_train[:200]
    y_train = y_train[:200]
    X_test = X_test[:200]
    y_test = y_test[:200]
    """

    ## Define Batch Size ##
    batch_size = 100

    ## Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    input_10_var = T.tensor4('10_inputs')
    support_var = T.tensor4('support_inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)

    network, network_for_deformation, weight_decay, network_transformed, output_10= build_cnn(input_var, support_var, batch_size)
    classification_network = build_linear(input_10_var, batch_size)

    saved_weights = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_hinge.npy")

    # Deformation matrix has 16 control points, coordiate number 2 * control points
    deformation_matrix_matrix = np.array(np.zeros((batch_size * 10, 2 * 16)), dtype = np.float32)

    network_saved_weights = np.array([deformation_matrix_matrix,] + [saved_weights[i] for i in range(saved_weights.shape[0])])

    lasagne.layers.set_all_param_values(network, network_saved_weights)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)

    predictions = lasagne.layers.get_output(network)

    predictions = T.reshape(predictions, (-1, 10, 10))

    predictions_deformation = lasagne.layers.get_output(network_for_deformation, deterministic = True)

    predictions_deformation_for_loss = lasagne.layers.get_output(network_for_deformation)

    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)
    
    output_10_value = lasagne.layers.get_output(output_10, deterministic = False)

    classfication_network_output = lasagne.layers.get_output(classification_network, deterministic = True)

    #loss_affine_before = lasagne.objectives.squared_error(predictions_deformation.clip(-20, 20), 20)

    loss_affine_before = -predictions_deformation

    loss_affine = loss_affine_before.mean() + weight_decay

    loss_affine = loss_affine_before.mean()

    loss = lasagne.objectives.multiclass_hinge_loss(predictions_deformation_for_loss, vanilla_target_var)
    loss = loss.mean()

    prediction_loss = lasagne.objectives.categorical_crossentropy(classfication_network_output, vanilla_target_var)

    prediction_loss_mean = prediction_loss.mean()

    # This is to use the rotation that the "gradient descent" think will give the highest score for each of the classes
    train_acc_1 = T.mean(T.eq(T.argmax(predictions_deformation_for_loss, axis = 1), vanilla_target_var), dtype = theano.config.floatX)

    # This is to use all the scores produces by all the rotations, compare them and get the highest one for each digit
    train_acc_2 = T.mean(T.eq(T.argmax(T.max(predictions, axis = 1), axis = 1), vanilla_target_var), dtype = theano.config.floatX)

    classification_acc = T.mean(T.eq(T.argmax(classfication_network_output, axis = 1), vanilla_target_var), dtype = theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)
    classification_params = lasagne.layers.get_all_params(classification_network, trainable=True)

    affine_params = params[0]
    model_params = params[1:]

    updates_affine = lasagne.updates.momentum(loss_affine, [affine_params], learning_rate = 0.3)

    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    updates_classification_model = lasagne.updates.adagrad(prediction_loss_mean, classification_params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1, 10, 10))

    test_prediction_deformation = lasagne.layers.get_output(network_for_deformation, deterministic=True)

    test_acc = T.mean(T.eq(T.argmax(T.max(test_prediction, axis=1), axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, support_var, vanilla_target_var], [loss, train_acc_1, train_acc_2], updates=updates_model)

    train_classification_model = theano.function([input_10_var, vanilla_target_var], [prediction_loss_mean, classification_acc], updates=updates_classification_model)

    train_affine_fn = theano.function([input_var, support_var], [loss_affine, loss_affine_before, transformed_images, weight_decay], updates=updates_affine)

    val_fn = theano.function([input_var, support_var, vanilla_target_var], [test_acc, output_10_value])
    val_classification = theano.function([input_10_var, vanilla_target_var], [prediction_loss_mean, classification_acc])



    # Finally, launch the training loop.
    cached_deformation_matrix = np.array(np.zeros((X_train.shape[0], 10, 2 * 16)), dtype = np.float32)
    weightsOfParams = lasagne.layers.get_all_param_values(network)
    weightsOfParams = np.load("../data/CNNForMNIST_tps_support_epoch99.npy")
    #weightsOfParams = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_Deformation_hinge_2000_script_run_reg_0_5_MNIST_em_new_epoch1500.npy")
    lasagne.layers.set_all_param_values(network, weightsOfParams)

    UseSupportSwitch = True
    EvaluationSwitch = True
    
    import amitgroup as ag
    import amitgroup.plot as gr

    class_support_none = np.ones((batch_size, 10, 40, 40), dtype=np.float32)
    if UseSupportSwitch:
        support_test_acc = 0
        class_support = np.load("../data/deformed_image_class_support.npy")
        class_support = np.array((class_support > 0.05), dtype = np.float32).reshape(1, 10, 40, 40)
        class_support = np.repeat(class_support, repeats=batch_size, axis = 0)
    else:
        class_support = class_support_none

    for epoch in range(num_epochs):
        start_time = time.time()
        transformed_images_list = []
        if (epoch % 200 == 0 or epoch + 1 == num_epochs) and EvaluationSwitch and epoch != 0:
            print ("Start Evaluating...")
            test_acc = 0
            test_batches = 0
            affine_test_batches = 0
            """
            if epoch + 1 == num_epochs:
                X_test = X_test_all
                y_test = y_test_all
            """
            cached_deformation_matrix_test = np.array(np.zeros((X_test.shape[0], 10, 2 * 16)), dtype = np.float32)

            # Find best affine transformation
            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape((batch_size, 1, 40, 40))
                affine_params.set_value(np.array(np.zeros((batch_size * 10, 2 * 16)), dtype = np.float32))
                for i in range(200):
                    weightsOfParams = lasagne.layers.get_all_param_values(network)
                    train_loss, train_loss_before, final_transformed_images, train_weight_decay_value = train_affine_fn(inputs, class_support)
                    if affine_test_batches == 0:
                        pass

                transformed_images_list.append(final_transformed_images)

                if affine_test_batches == 0:
                    pass
                    np.save(os.environ['TMP'] + "/clutter_deformed_images_epoch_%d.npy" %epoch, final_transformed_images)
                    np.save(os.environ['TMP'] + "/clutter_deformed_images_original_epoch_%d.npy" %epoch, inputs)

                cached_deformation_matrix_test[index] = weightsOfParams[0].reshape((-1, 10, 2 * 16))

                affine_test_batches += 1
                print(affine_test_batches)
            error_image = []
            error_original_image = []
            error_transformed_image = []
            error_prediction = []
            correct_prediction = []

            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
                inputs, targets, index = batch
                affine_params.set_value(cached_deformation_matrix_test[index].reshape((-1, 2 * 16)))
                inputs = inputs.reshape((batch_size, 1, 40, 40))
                acc, vector_10_output= val_fn(inputs, class_support, targets)
                loss_value, acc = val_classification(vector_10_output, targets)
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
        
        print("Starting training...")
        train_err = 0
        train_acc_sum = 0
        train_batches = 0
        affine_train_batches = 0

        if epoch % 200 == 0:
            print("inside")
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            batch_loss = 0
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape((batch_size, 1, 40, 40))
                
                affine_params.set_value(np.array(np.zeros((batch_size * 10, 2 * 16)), dtype = np.float32))
                final_transformed_images = None
                for i in range(200):
                    weightsOfParams = lasagne.layers.get_all_param_values(network)
                    train_loss, train_loss_before, final_transformed_images, weight_decay = train_affine_fn(inputs, class_support)
                cached_deformation_matrix[index] = weightsOfParams[0].reshape((-1, 10, 2 * 16))
                affine_train_batches += 1
                batch_loss += np.mean(train_loss_before)
                print(affine_train_batches)
            print(batch_loss / affine_train_batches)
            print (time.time() - start_time)

        if 1:
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets, index = batch
                affine_params.set_value(cached_deformation_matrix[index].reshape((-1, 2 * 16)))
                inputs = inputs.reshape((batch_size, 1, 40, 40))
                acc, vector_10_output = val_fn(inputs, class_support, targets)
                train_loss_value, train_acc_value = train_classification_model(vector_10_output, targets)
                train_err += train_loss_value
                train_acc_sum += train_acc_value
                train_batches += 1
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc:\t\t{:.6f}".format(train_acc_sum / train_batches))

        if epoch % 100 == 0: 
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            np.save("../data/new_support_trial_epoch%d.npy" %epoch, weightsOfParams)


if __name__ == '__main__':
    main()