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
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc1, p=.5),
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=10,
            )

    network_transformed = lasagne.layers.ReshapeLayer(network_transformed, (-1, 10, 10, 40, 40))

    fc2_selected = SelectLayer(fc2, 10)
    
    weight_decay_layers = {fc1:0.0, fc2:0.002}
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
        yield inputs[excerpt], targets[excerpt], indices[start_idx:start_idx+batchsize]


def extend_image(inputs, size = 40):
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    margin_size = (40 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images


def main(model='mlp', num_epochs=1):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class) 
    print("Using all the training data") 
    
    ## Load Data##
    X_train, y_train, X_test, y_test = load_data("/mnistROT.npy", "/mnistROTLabel.npy", "/mnistROTTEST.npy", "/mnistROTLABELTEST.npy", "ROT_MNIST")
    
    X_train = extend_image(X_train, 40)
    X_test = extend_image(X_test, 40)


    ## Define Batch Size ##
    batch_size = 100
 
    ## Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)
    
    network, network_for_rotation, weight_decay, network_transformed = build_cnn(input_var, batch_size)
    network_exhaustive, _ = build_rotation_cnn(input_var)
    
    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    saved_weights = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_hinge.npy")

    affine_matrix_matrix = np.array(np.zeros((batch_size * 10,)), dtype = np.float32)
    
    network_saved_weights = np.array([affine_matrix_matrix,] + [saved_weights[i] for i in range(saved_weights.shape[0])])
    
    lasagne.layers.set_all_param_values(network, network_saved_weights)
    lasagne.layers.set_all_param_values(network_exhaustive, saved_weights)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)

    predictions = lasagne.layers.get_output(network)

    predictions = T.reshape(predictions, (-1, 10, 10))
    
    predictions_rotation = lasagne.layers.get_output(network_for_rotation, deterministic = True)
    
    predictions_rotation_for_loss = lasagne.layers.get_output(network_for_rotation)
    
    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)

    predictions_exhaustive = lasagne.layers.get_output(network_exhaustive, deterministic = True)

    predictions_exhaustive = T.reshape(predictions_exhaustive, (nRotation, -1, 10))
    
    predictions_exhaustive_degree = T.argmax(predictions_exhaustive, 0) * 45.0

    predictions_exhaustive = T.max(predictions_exhaustive, 0)
    
    
    #loss_affine_before = lasagne.objectives.squared_error(predictions_rotation.clip(-20, 3), 3) + lasagne.objectives.squared_error(predictions_rotation.clip(3, 20), 20)
    loss_affine_before = lasagne.objectives.squared_error(predictions_rotation.clip(-20, 20), 20)
    #loss_affine_exhaustive_before = lasagne.objectives.squared_error(predictions_exhaustive.clip(-20, 3), 3) + lasagne.objectives.squared_error(predictions_exhaustive.clip(3, 20), 20)
    loss_affine_exhaustive_before = lasagne.objectives.squared_error(predictions_exhaustive.clip(-20, 20), 20)

    loss_affine = loss_affine_before.mean()
    loss_affine_exhaustive = loss_affine_exhaustive_before.mean()

    loss = lasagne.objectives.multiclass_hinge_loss(predictions_rotation_for_loss, vanilla_target_var)
    loss = loss.mean() + weight_decay

    # This is to use the rotation that the "gradient descent" think will give the highest score for each of the classes
    train_acc_1 = T.mean(T.eq(T.argmax(predictions_rotation, axis = 1), vanilla_target_var), dtype = theano.config.floatX)
    
    # This is to use all the scores produces by all the rotations, compare them and get the highest one for each digit
    train_acc_2 = T.mean(T.eq(T.argmax(T.max(predictions, axis = 1), axis = 1), vanilla_target_var), dtype = theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    

    affine_params = params[0]
    model_params = params[1:]
    
    # updates_affine = lasagne.updates.sgd(loss_affine, [affine_params], learning_rate = 10)
    d_loss_wrt_params = T.grad(loss_affine, [affine_params])
    
    updates_affine = OrderedDict()

    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 500 * grad

    
    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1, 10, 10))
    
    test_prediction_rotation = lasagne.layers.get_output(network_for_rotation, deterministic=True)
    
    test_acc = T.mean(T.eq(T.argmax(T.max(test_prediction, axis=1), axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    loss_test_affine = lasagne.objectives.squared_error(test_prediction_rotation.clip(-20, 3), 3) + lasagne.objectives.squared_error(test_prediction_rotation.clip(3, 20), 20)

    loss_test_affine = loss_test_affine.mean()

    update_affine_test = lasagne.updates.adagrad(loss_test_affine, [affine_params], learning_rate = 10)
    
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var], [loss,train_acc_1, train_acc_2], updates=updates_model)
    
    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before], updates=updates_affine)

    get_affine_exhaustive_fn = theano.function([input_var], [loss_affine_exhaustive, loss_affine_exhaustive_before, predictions_exhaustive_degree])
    
    val_affine_fn = theano.function([input_var], [loss_test_affine], updates=update_affine_test)
    
    val_fn = theano.function([input_var, vanilla_target_var], test_acc)
    

    # Finally, launch the training loop.
    # We iterate over epochs:
    cached_affine_matrix = np.array(np.zeros((X_train.shape[0], 10,)), dtype = np.float32)
    cached_affine_matrix_test = np.array(np.zeros((X_test.shape[0], 10,)), dtype = np.float32)
    for epoch in range(num_epochs):
        """        
        if epoch % 400 == -1:
            print ("Start Evaluating...")
            test_acc = 0
            test_batches = 0
            affine_test_batches = 0
            if epoch != 0:
                for batch in iterate_minibatches(X_test, y_test, affine_network_size, shuffle=False):
                    inputs, targets, index = batch
                    affine_params_newnet.set_value(cached_affine_matrix_test[index].reshape(-1,))
                    inputs = inputs.reshape(affine_network_size, 1, 40, 40)
                    fake_targets = np.zeros((affine_network_size, 10))
                    fake_targets[:, ] = np.arange(10)
                    fake_targets = np.array(fake_targets.reshape((affine_network_size * 10,)), dtype = np.int32)
                    for i in range(100):
                        train_loss = val_affine_fn(inputs, fake_targets)
                    weightsOfParams = lasagne.layers.get_all_param_values(network_affine)
                    cached_affine_matrix_test[index] = weightsOfParams[0].reshape(-1, 10)
                    affine_test_batches += 1
                    print(affine_test_batches)

            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
                inputs, targets, index = batch
                affine_params.set_value(cached_affine_matrix_test[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 1, 40, 40)
                transformed_image_res = None
                fake_targets = np.zeros((batch_size, 10))
                fake_targets[:, ] = np.arange(10)
                fake_targets = np.array(fake_targets.reshape((batch_size * 10,)), dtype = np.int32)
                acc = val_fn(inputs, targets, fake_targets)
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
        """

        print("Starting training...")
        train_err = 0
        train_acc_sum = 0
        train_batches = 0
        affine_train_batches = 0

        if epoch % 200 == 0:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            ## Assign weights to network rotation##
            lasagne.layers.set_all_param_values(network_exhaustive, weightsOfParams[1:])
            batch_loss = 0
            batch_loss_exhaustive = 0
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                start_time = time.time()
                print(affine_train_batches)
                inputs, targets, index = batch
                # affine_params.set_value(cached_affine_matrix[index].reshape(-1,))
                #rotated_inputs = rotateImage_batch(inputs, nRotation).reshape(batch_size * nRotation, 1, 40, 40)
                inputs = inputs.reshape(batch_size, 1, 40, 40)
                
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 1
                eachDegree = 360.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size * 10) * eachDegree * j, dtype = np.float32))
                    train_loss_before = None
                    for i in range(1):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        if i == 19:
                            print(weightsOfParams[0].reshape(batch_size, 10)[0])
                        # train_loss, train_loss_before, gradient_loss, _ = train_affine_fn(inputs)
                        train_loss, train_loss_before = train_affine_fn(inputs)
                        if i == 19:
                            print(train_loss_before.reshape(batch_size, 10)[0])
                            lasagne.layers.set_all_param_values(network, weightsOfParams)
                            # train_loss, train_loss_before, gradient_loss, _ = train_affine_fn(inputs)
                            train_loss, train_loss_before = train_affine_fn(inputs)
                            print(train_loss_before.reshape(batch_size, 10)[0])
                        #print(train_loss, targets[0])
                        #print(gradient_loss.shape, gradient_loss.reshape(-1, 10)[0])
                        #print(train_loss_before[0])
                        #print(weightsOfParams[0].reshape(-1, 10)[0])
                    affine_params_all.append(np.array(weightsOfParams[0].reshape(1, batch_size, 10)))
                    train_loss_before_all.append(train_loss_before.reshape(1, batch_size, 10))
                    #print("---------------------------------------------------------------------")

                train_loss_before_all = np.vstack(train_loss_before_all)
                affine_params_all = np.vstack(affine_params_all)
                print(affine_params_all[:, 0, :])
                print(train_loss_before_all[:, 0, :])
                
                train_loss_before_all_reshape = train_loss_before_all.reshape(searchCandidate, -1)
                affine_params_all_reshape = affine_params_all.reshape(searchCandidate, -1)

                train_arg_min = np.argmin(train_loss_before_all_reshape, axis = 0)

                affine_params_all_reshape = affine_params_all_reshape[train_arg_min, np.arange(train_arg_min.shape[0])]
                print(affine_params_all_reshape.reshape(batch_size, 10)[0])
                affine_params.set_value(affine_params_all_reshape)
                # final_loss_all, final_loss_before, _, transformed_image_res = train_affine_fn(inputs)
                final_loss_all, final_loss_before = train_affine_fn(inputs)
                affine_params_all_reshape = affine_params_all_reshape.reshape(batch_size, 10)

                train_loss_before_all = np.min(train_loss_before_all, axis = 0)
                #train_loss_exhaustive, train_loss_exhaustive_before, exhaustive_degree = get_affine_exhaustive_fn(rotated_inputs)
                #print(exhaustive_degree[0])
                print(affine_params_all_reshape[0])
                """
                print(train_loss, train_loss_exhaustive, targets[0])
                print(gradient_loss.shape, gradient_loss.reshape(-1, 10)[0])
                print(train_loss_before[0], train_loss_exhaustive_before[0])
                weightsOfParams = lasagne.layers.get_all_param_values(network)
                print(weightsOfParams[0].reshape(-1, 10)[0])
                """
                #print(train_loss_before_all[0], train_loss_exhaustive_before[0], final_loss_before[0])
                #print(np.mean(train_loss_before_all), train_loss_exhaustive)
                #unrotated_image = np.array([[rotateImage(inputs[i], exhaustive_degree[i][j]) for j in range(10)] for i in range(100)])
                print("====================================")
                cached_affine_matrix[index] = weightsOfParams[0].reshape(-1, 10)
                affine_train_batches += 1
                batch_loss += final_loss_all
                #batch_loss_exhaustive += train_loss_exhaustive
                if affine_train_batches == 1:
                    np.save("transformed_res.npy", transformed_image_res)
                    #np.save("unrotated_image.npy", unrotated_image)
                    np.save("original_res.npy", inputs.reshape(batch_size, 40, 40))
                    break
            print(batch_loss / affine_train_batches)
            #print(batch_loss_exhaustive / affine_train_batches)
            print (time.time() - start_time)
        """
        if 1:
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets, index = batch
                affine_params.set_value(cached_affine_matrix[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 1, 40, 40)
                transformed_image_res = None
                fake_targets = np.zeros((batch_size, 10))
                fake_targets[:, ] = np.arange(10)
                fake_targets = np.array(fake_targets.reshape((batch_size * 10,)), dtype = np.int32)
                train_loss_value, train_acc_value = train_model_fn(inputs, targets, fake_targets)
                train_err += train_loss_value
                train_acc_sum += train_acc_value
                train_batches += 1
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc:\t\t{:.6f}".format(train_acc_sum / train_batches))

        
    weightsOfParams = lasagne.layers.get_all_param_values(network)
    np.save("../data/mnist_CNN_params_drop_out_Chi_2017_ROT_hinge_2000_em_final.npy", weightsOfParams)
    """


if __name__ == '__main__':
    main()
