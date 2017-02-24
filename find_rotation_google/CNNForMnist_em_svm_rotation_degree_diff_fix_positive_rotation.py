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
from lasagne.layers import BatchNormLayer
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

    norm0 = BatchNormLayer(input_transformed)
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



def iterate_minibatches(inputs, targets, batchsize, degrees = None, shuffle=False):
    assert len(inputs) == len(targets)

    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if(start_idx + batchsize < len(inputs)):
            if degrees is not None:
                yield inputs[excerpt], targets[excerpt], degrees[excerpt], indices[start_idx:start_idx+batchsize]
            else:
                yield inputs[excerpt], targets[excerpt], indices[start_idx:start_idx+batchsize]
        else:
            num_needed = start_idx + batchsize - len(inputs)
            if degrees is not None:
                yield np.vstack([inputs[excerpt], inputs[indices[0:num_needed]]]), np.concatenate([targets[excerpt], targets[indices[0:num_needed]]]), np.concatenate([degrees[excerpt], degrees[indices[0:num_needed]]]), np.concatenate([indices[start_idx: start_idx + batchsize], indices[0:num_needed]])
            else:
                yield np.vstack([inputs[excerpt], inputs[indices[0:num_needed]]]), np.concatenate([targets[excerpt], targets[indices[0:num_needed]]]), np.concatenate([indices[start_idx: start_idx + batchsize], indices[0:num_needed]])

                

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
    X_train, y_train, X_test, y_test = load_data("/X_train_rotated.npy", "/Y_train_rotated.npy", "/X_test_rotated.npy", "/Y_test_rotated.npy", dataType = "test")
    # X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    # X_train = extend_image(X_train)
    #X_test = extend_image(X_test)
    X_train_degree = np.load("../car/Z_degree_train_rotated.npy")
    X_test_degree = np.load("../car/Z_degree_test_rotated.npy")

    ## Define Batch Size ##
    batch_size = 100
 
    ## Define nRotation for exhaustive search ##
    nRotation = 8

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)
    
    network, network_transformed = build_cnn(input_var, batch_size)
    
    #saved_weights = np.load("../data/google_car_CNN_params_drop_out_Chi_2017_hinge.npy")
    saved_weights = np.load("../data/google_car_CNN_params_drop_out_Chi_2017_hinge_class_2_fake.npy")
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
    loss = loss.mean()

    # This is to use all the scores produces by all the rotations, compare them and get the highest one for each digit
    train_acc = T.mean(T.eq((predictions > 0), vanilla_target_var), dtype = theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    affine_params = params[0]
    model_params = params[1:]
    
    # updates_affine = lasagne.updates.sgd(loss_affine, [affine_params], learning_rate = 10)
    d_loss_wrt_params = T.grad(loss_affine, [affine_params])
    
    updates_affine = OrderedDict()

    for param, grad in zip([affine_params], d_loss_wrt_params):
        updates_affine[param] = param - 20 * grad

    
    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    #updates_model = lasagne.updates.sgd(loss, params, learning_rate=0.00001)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_prediction = T.reshape(test_prediction, (-1,))
    
    test_acc = T.mean(T.eq((test_prediction > 0), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var], [loss,train_acc, transformed_images], updates=updates_model)
    
    train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before, predictions_rotation, transformed_images] + d_loss_wrt_params, updates=updates_affine)
    
    get_transformed_image = theano.function([input_var], [transformed_images, test_prediction])

    val_fn = theano.function([input_var, vanilla_target_var], test_acc)

    # Finally, launch the training loop.
    # We iterate over epochs:
    cached_affine_matrix = np.array(np.zeros((X_train.shape[0],)), dtype = np.float32)
    for epoch in range(num_epochs):
        start_time = time.time()

        if 1:
            print ("Start Evaluating...")
            test_acc = 0
            degree_diff = 0
            test_batches = 0
            affine_test_batches = 0
            # Find best rotation
            cached_affine_matrix_test = np.array(np.zeros((X_test.shape[0],)), dtype = np.float32)
            all_degree = []
            all_target = []
            all_transformed = []
            all_current_prediction = []
            all_correct_prediction = []
            for batch in iterate_minibatches(X_test, y_test, batch_size, X_test_degree, shuffle=False):
                inputs, targets, correct_degree, index = batch
                inputs = inputs.reshape(batch_size, 3, 68, 68)
                train_loss_before_all = []
                affine_params_all = []
                searchCandidate = 8
                eachDegree = 360.0 / searchCandidate
                for j in range(searchCandidate):
                    affine_params.set_value(np.array(np.ones(batch_size) * eachDegree * j, dtype = np.float32))
                    for i in range(20):
                        weightsOfParams = lasagne.layers.get_all_param_values(network)
                        train_affine_fn_result = train_affine_fn(inputs)
                        train_loss, train_loss_before, _ = train_affine_fn_result[:3]
                
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
                car_degree = correct_degree[targets == 1] % 360
                find_degree = -cached_affine_matrix_test[index][targets == 1]%360
                degree_diff_array = 180 - np.abs(180 - (car_degree - find_degree) % 360)
                #degree_diff_array[degree_diff_array > 90] = 180 - degree_diff_array[degree_diff_array > 90]
                degree_diff += np.sum(degree_diff_array)
                affine_test_batches += 1
                if affine_test_batches == 1:
                    print(degree_diff_array)
                print(affine_test_batches)

                affine_params.set_value(affine_params_all_reshape.reshape(-1,))
                transformed_images, current_prediction = get_transformed_image(inputs)
                all_current_prediction.append(current_prediction)

                affine_params.set_value(-np.array(correct_degree, dtype = np.float32).reshape(-1, ))
                correct_transformed_images, correct_prediction = get_transformed_image(inputs)
                all_correct_prediction.append(correct_prediction)
        
                all_transformed.append(transformed_images)
                all_degree.append(correct_degree)
                all_target.append(targets)

            all_degree = np.concatenate(all_degree)
            all_transformed = np.concatenate(all_transformed)
            all_current_prediction = np.concatenate(all_current_prediction)
            all_correct_prediction = np.concatenate(all_correct_prediction)
            all_target = np.concatenate(all_target)

            all_degree = all_degree[:X_test.shape[0]]
            all_target = all_target[:X_test.shape[0]]
            all_transformed = all_transformed[:X_test.shape[0]]
            all_correct_prediction = all_correct_prediction[:X_test.shape[0]]
            all_current_prediction = all_current_prediction[:X_test.shape[0]]



            all_transformed = all_transformed[all_target == 1]
            all_current_prediction = all_current_prediction[all_target == 1]
            all_correct_prediction = all_correct_prediction[all_target == 1]


            final_car_degree = all_degree[all_target == 1] % 360

            print("test car number: ", final_car_degree.shape)
            final_find_car_degree = -cached_affine_matrix_test[all_target == 1] % 360
            degree_diff_array = 180 - np.abs(180 - (final_car_degree - final_find_car_degree) % 360)
            print("Larger than 150: ", np.sum(degree_diff_array >= 150))
            print("Smaller than 15: ", np.sum(degree_diff_array <= 15))
            valid_inputs = X_test[y_test == 1]
            np.save("hard_cars.npy", valid_inputs[degree_diff_array >= 150])
            np.save("transformed_hard_cars.npy", all_transformed[degree_diff_array >= 150])
            np.save("degree_diff_array_2_class.npy", degree_diff_array)

            print("Hard cars - our orientation prediction", all_current_prediction[degree_diff_array >= 150])
            print("Hard cars - correct orientation prediciton", all_correct_prediction[degree_diff_array >= 150])

            print(final_car_degree[degree_diff_array >= 150])
            print(final_find_car_degree[degree_diff_array >= 150])

            # Hard inputs are the following:
            hard_input = X_test[y_test == 1][degree_diff_array >= 150]


            print("Evaluation average degree difference: ", degree_diff / X_test[y_test == 1].shape[0], np.mean(degree_diff_array))
            degree_diff_array[degree_diff_array > 90] = 180 - degree_diff_array[degree_diff_array > 90]
            print("Evaluation average degree difference without 180: ", np.mean(degree_diff_array))
            

            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle = True):
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
        degree_diff = 0
        weightsOfParams = lasagne.layers.get_all_param_values(network)
        batch_loss = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, X_train_degree, shuffle=True):
            inputs, targets, correct_degree, index = batch
            inputs = inputs.reshape(batch_size, 3, 68, 68)
            
            train_loss_before_all = []
            affine_params_all = []
            searchCandidate = 8
            eachDegree = 360.0 / searchCandidate
            print(np.max(inputs), np.min(inputs))
            for j in range(searchCandidate):
                affine_params.set_value(np.array(np.ones(batch_size) * eachDegree * j, dtype = np.float32))
                #affine_params.set_value(np.array(np.ones(batch_size) * 119.0, dtype = np.float32))
                for i in range(20):
                    weightsOfParams = lasagne.layers.get_all_param_values(network)
                    train_affine_result = train_affine_fn(inputs)
                    train_loss, train_loss_before, all_predictions, transformed_images = train_affine_result[:4]
                    #print("Current Degree: ", weightsOfParams[0][:20])
                    #print("Correct Degree: ", degree[:20])
                    #print("Current Gradient: ", train_affine_result[4:24])
                    #print("Current Loss: ", train_loss_before[:20])
                    #print("Current Label: ", targets[:20])
                    #print("Current Mean Loss: ", train_loss)
                    #np.save("two_images.npy", np.vstack([inputs[0][:,10:58, 10:58], transformed_images[0]]))
                    #exit()
                    #time.sleep(5)
                
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
            
            train_loss_before_all = np.min(train_loss_before_all, axis = 0)
            
            cached_affine_matrix[index] = affine_params_all_reshape.reshape(-1,)
            cached_affine_matrix[index[targets == 1]] = -correct_degree[targets == 1] % 360
            car_degree = correct_degree[targets == 1] % 360
            find_degree = -cached_affine_matrix[index][targets == 1] % 360
            degree_diff_array = 180 - np.abs(180 - (car_degree - find_degree) % 360)
            #degree_diff_array[degree_diff_array > 90] = 180 - degree_diff_array[degree_diff_array > 90]
            degree_diff += np.sum(degree_diff_array)
            affine_train_batches += 1
            batch_loss += np.mean(train_loss_before_all)
            print(affine_train_batches)
        print(batch_loss / affine_train_batches)
        print (time.time() - start_time)
        print("Train average degree diff: ", degree_diff / X_train[y_train == 1].shape[0])

        if 1:
            for batch in iterate_minibatches(X_train, y_train, batch_size, X_train_degree, shuffle=True):
                inputs, targets, degree, index = batch
                affine_params.set_value(cached_affine_matrix[index].reshape(-1,))
                inputs = inputs.reshape(batch_size, 3, 68, 68)
                train_loss_value, train_acc_value_1, transformed_image_res = train_model_fn(inputs, targets)
                train_err += train_loss_value
                train_acc_sum_1 += train_acc_value_1
                train_batches += 1
                if train_batches == 1:
                    np.save("original_image.npy", inputs)
                    np.save("transformed_image.npy", transformed_image_res)
                    print("original degree: ", degree[targets == 1] % 360)
                    print("find degree: ", -cached_affine_matrix[index][targets == 1] % 360)
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc 1:\t\t{:.6f}".format(train_acc_sum_1 / train_batches))
        
        if epoch % 10 == 0:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            np.save("../data/google_earth_em_new_batch_run_epoch_%d_degree_diff_fix_2.npy" %epoch, weightsOfParams)

        
if __name__ == '__main__':
    main()
