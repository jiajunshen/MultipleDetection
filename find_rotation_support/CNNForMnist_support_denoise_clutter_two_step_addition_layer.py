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
from rotationMatrixLayer import RotationTransformationLayer
from selectLayer import SelectLayer
from CNNForMnist_Rotation_Net import build_cnn as build_rotation_cnn
from CNNForMnist_Rotation_Net import rotateImage_batch
from CNNForMnist_Rotation_Net import rotateImage
from collections import OrderedDict

def build_cnn(input_var=None, batch_size = None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 40, 40),
                                        input_var=input_var)

    repeatInput = Repeat(network, 10)

    network = lasagne.layers.ReshapeLayer(repeatInput, (-1, 1, 40, 40))

    network_transformed = RotationTransformationLayer(network, batch_size * 10)

    network = Conv2DLayer(
            network_transformed, num_filters=32, filter_size=(5, 5),
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

    network_transformed = lasagne.layers.ReshapeLayer(network_transformed, (-1, 10, 40, 40))

    fc2_selected = SelectLayer(fc2, 10)

    weight_decay_layers = {network_transformed:0.0}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc2, fc2_selected, l2_penalty, network_transformed

def build_cnn_step_two(input_var=None, batch_size = None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 40, 40),
                                        input_var=input_var)

    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
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

    #network = lasagne.layers.ReshapeLayer(network, (-1, 10, 7, 7))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    fc1_reshape = lasagne.layers.ReshapeLayer(fc1, (-1, 2560))
    
    fc3  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc1_reshape, p=.5),
            #network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            )
    
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc3, p=.5),
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


def main(model='mlp', num_epochs=500):
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
    y_test = y_test

    X_train = X_train
    y_train = y_train
    X_test = X_test
    y_test = y_test


    ## Define Batch Size ##
    batch_size = 100

    ## Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    two_step_input = T.tensor4("two_step_inputs")
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)

    network, network_for_deformation, weight_decay, network_transformed = build_cnn(input_var, batch_size)
    two_step_classifcation_network = build_cnn_step_two(two_step_input, batch_size)

    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")

    saved_weights = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_hinge.npy")

    affine_matrix_matrix = np.array(np.zeros((batch_size * 10,)), dtype = np.float32)
    
    network_saved_weights = np.array([affine_matrix_matrix,] + [saved_weights[i] for i in range(saved_weights.shape[0])])
    
    lasagne.layers.set_all_param_values(network, network_saved_weights)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)

    predictions = lasagne.layers.get_output(network)

    predictions = T.reshape(predictions, (-1, 10, 10))

    predictions_deformation = lasagne.layers.get_output(network_for_deformation, deterministic = True)

    predictions_deformation_for_loss = lasagne.layers.get_output(network_for_deformation)

    two_step_prediction_train = lasagne.layers.get_output(two_step_classifcation_network, deterministic = False)
    two_step_prediction_test = lasagne.layers.get_output(two_step_classifcation_network, deterministic = True)

    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)

    #loss_affine_before = lasagne.objectives.squared_error(predictions_deformation.clip(-20, 20), 20)

    loss_affine_before = lasagne.objectives.squared_error(predictions_deformation.clip(-20, 20), 20)

    loss_affine = loss_affine_before.mean()

    loss = lasagne.objectives.multiclass_hinge_loss(predictions_deformation_for_loss, vanilla_target_var)
    loss = loss.mean()

    two_step_classification_loss = lasagne.objectives.categorical_crossentropy(two_step_prediction_train, vanilla_target_var)

    two_step_classification_loss_mean = two_step_classification_loss.mean()

    # This is to use the rotation that the "gradient descent" think will give the highest score for each of the classes
    train_acc_1 = T.mean(T.eq(T.argmax(predictions_deformation_for_loss, axis = 1), vanilla_target_var), dtype = theano.config.floatX)

    # This is to use all the scores produces by all the rotations, compare them and get the highest one for each digit
    train_acc_2 = T.mean(T.eq(T.argmax(T.max(predictions, axis = 1), axis = 1), vanilla_target_var), dtype = theano.config.floatX)

    two_step_acc = T.mean(T.eq(T.argmax(two_step_prediction_test, axis = 1), vanilla_target_var), dtype = theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)
    two_step_classifcation_network_params = lasagne.layers.get_all_params(two_step_classifcation_network)

    affine_params = params[0]
    model_params = params[1:]
    
    # updates_affine = lasagne.updates.sgd(loss_affine, [affine_params], learning_rate = 10)
    d_loss_wrt_params = T.grad(loss_affine, [affine_params])
    
    updates_affine = OrderedDict()

    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 1000 * grad

    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    update_two_step_classification_network = lasagne.updates.adagrad(two_step_classification_loss_mean, two_step_classifcation_network_params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1, 10, 10))

    test_prediction_deformation = lasagne.layers.get_output(network_for_deformation, deterministic=True)

    test_acc = T.mean(T.eq(T.argmax(T.max(test_prediction, axis=1), axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var], [loss, train_acc_1, train_acc_2], updates=updates_model)

    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before, transformed_images, transformed_images], updates=updates_affine)

    train_two_step_model_fn = theano.function([two_step_input, vanilla_target_var], [two_step_classification_loss_mean, two_step_acc], updates=update_two_step_classification_network)

    val_fn = theano.function([input_var, vanilla_target_var], [test_acc, transformed_images])

    val_two_step_fn = theano.function([two_step_input, vanilla_target_var], [two_step_acc,T.argmax(two_step_prediction_test, axis = 1)] )


    # Finally, launch the training loop.
    cached_affine_matrix = np.array(np.zeros((X_train.shape[0], 10,)), dtype = np.float32)
    weightsOfParams = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_ROT_hinge_2000_em_new_script_run_3_epoch_1300.npy")
    lasagne.layers.set_all_param_values(network, weightsOfParams)
    transformed_images_train_list = []
    transformed_images_list = []
    UseSupportSwitch = True
    EvaluationSwitch = True

    for epoch in range(num_epochs):
        start_time = time.time()
        if (epoch % 50 == 0 or epoch + 1 == num_epochs) and EvaluationSwitch:
            transformed_images_list = []
            print ("Start Evaluating...")
            test_acc = 0
            test_batches = 0
            affine_test_batches = 0
            """
            if epoch + 1 == num_epochs:
                X_test = X_test_all
                y_test = y_test_all
            """
            cached_affine_matrix_test = np.array(np.zeros((X_test.shape[0], 10,)), dtype = np.float32)
            
            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape(batch_size, 1, 40, 40)
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 6
                eachDegree = 360.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j, dtype = np.float32))
                    for i in range(10):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_loss, train_loss_before, _, _ = train_affine_fn(inputs)
                
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
                inputs = inputs.reshape(batch_size, 1, 40, 40)
                acc, transformed_images_result = val_fn(inputs, targets)
                transformed_images_list.append(transformed_images_result)
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))



        import amitgroup as ag
        import amitgroup.plot as gr
        if UseSupportSwitch:
            support_test_acc = 0
            class_support = np.load("/hdd/Documents/tmp/rotation_image_support_new.npy")
            class_support = (class_support > 0.1)
            transformed_images_list_support = np.vstack(transformed_images_list)
            for i in range(10):
                transformed_images_list_support[:,i] *= np.array(class_support[i], dtype = np.float32)
            error_image = []
            error_original_image = []
            error_transformed_image = []
            error_prediction = []
            correct_prediction = []
            for i in range(transformed_images_list_support.shape[0] // batch_size):
                original_image = X_test[i * batch_size: (i + 1) * batch_size]
                targets = y_test[i * batch_size: (i + 1) * batch_size]
                acc, pred= val_two_step_fn(transformed_images_list_support[i * batch_size : (i + 1) * batch_size].reshape(batch_size * 10, 1, 40, 40), targets)
                error_image.append(transformed_images_list_support[i * batch_size : (i + 1) * batch_size].reshape(batch_size, 10, 40, 40)[pred != targets])
                error_original_image.append(original_image[pred != targets])
                error_transformed_image.append(np.vstack(transformed_images_list)[i * batch_size : (i + 1) * batch_size].reshape(batch_size, 10, 40, 40)[pred != targets])
                error_prediction.append(pred[pred!=targets])

                support_test_acc += acc
                #print(targets)
                #gr.images(np.array([transformed_images_list[i * batch_size : (i + 1) * batch_size].reshape(batch_size, 10, 40, 40)[j, targets[j]] for j in range(batch_size)]))
            support_test_acc /= (len(y_test) / 100.0)
            print("Final result support: ")
            print("  test accuracy:\t\t{:.2f} %".format(support_test_acc * 100))
            np.save("./error_image_rotation.npy", np.vstack(error_image))
            np.save("./error_prediction_rotation.npy", np.concatenate(error_prediction))
            np.save("./error_original_image_rotation.npy", np.vstack(error_original_image))
            np.save("./error_transformed_image_rotation.npy", np.vstack(error_transformed_image))
         
        if epoch % 50 == 0:
            transformed_images_train_list = []
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            batch_loss = 0
            affine_train_batches = 0
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets, index = batch
                inputs = inputs.reshape(batch_size, 1, 40, 40)
                
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 6
                eachDegree = 360.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j, dtype = np.float32))
                    for i in range(10):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_loss, train_loss_before, _, _ = train_affine_fn(inputs)
                    
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)
                
                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)
                
                train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)

                affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                
                train_loss_before_all = np.min(train_loss_before_all, axis = 0)
                
                cached_affine_matrix[index] = affine_params_all_reshape.reshape(-1, 10)
                affine_train_batches += 1
                batch_loss += np.mean(train_loss_before_all)
                print(affine_train_batches) 
            print(batch_loss / affine_train_batches)
            print(time.time() - start_time)

            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle = False):
                inputs, targets, index = batch
                affine_params.set_value(cached_affine_matrix[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 1, 40, 40)
                acc, transformed_images_result = val_fn(inputs, targets)
                transformed_images_train_list.append(transformed_images_result)

        if 1:
            class_support = np.load("/hdd/Documents/tmp/rotation_image_support_new.npy")
            class_support = (class_support > 0.1)
            transformed_images_list_support = np.vstack(transformed_images_train_list)
            train_err = 0
            train_batches = 0
            train_acc_sum = 0
            for i in range(10):
                transformed_images_list_support[:, i] *= np.array(class_support[i], dtype = np.float32)
            transformed_images_list_support = transformed_images_list_support.reshape(-1, 10, 40, 40)
            transformed_images_list_support = np.array(transformed_images_list_support, dtype = np.float32)
            for batch in iterate_minibatches(transformed_images_list_support, y_train, batch_size, shuffle=True):
                inputs, targets, index = batch
                inputs = inputs.reshape(-1, 1, 40, 40)
                train_loss_value, train_acc_value = train_two_step_model_fn(inputs, targets)
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
            np.save("../data/new_support_trial_epoch_rotation_%d.npy" %epoch, weightsOfParams)


if __name__ == '__main__':
    main()
