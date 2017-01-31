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
from rotationMatrixLayer import RotationMatrixLayer

def one_vs_all_hinge_loss(predictions, targets, delta = 1):
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    corrects = T.reshape(predictions[targets.nonzero()], (-1, 1))
    rest = theano.tensor.reshape(predictions[(1-targets).nonzero()],
                                 (-1, num_cls-1))
    rest = rest - corrects + delta
    rest = theano.tensor.sum(rest, axis = 1)
    return theano.tensor.nnet.relu(rest)


def build_cnn(input_var=None, batch_size = 100, constant_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(batch_size, 1, 40, 40),
                                        input_var=input_var)

    repeatInput = Repeat(network, 10)

    network = lasagne.layers.ReshapeLayer(repeatInput, (-1, 1, 40, 40))
    
    side_network = lasagne.layers.InputLayer(shape=(1, 1),
                                             input_var=constant_var)

    side_network = RotationMatrixLayer(side_network, n = batch_size * 10)

    network_transformed = lasagne.layers.TransformerLayer(network, side_network)                               
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = Conv2DLayer(
            network_transformed, num_filters=32, filter_size=(5, 5),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

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


    # fc2 = lasagne.layers.ReshapeLayer(fc2, (-1, 10, 10))

    network_transformed = lasagne.layers.ReshapeLayer(network_transformed, (-1, 10, 40, 40))
    
    weight_decay_layers = {fc1:0.0, fc2:0.002}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc2, l2_penalty, network_transformed, side_network



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


def main(model='mlp', num_epochs=2000):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class) 
    print("Using all the training data") 
    
    batch_size = 100 
    #X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test_rotated.npy", "/Y_test_rotated.npy")
    X_train, y_train, X_test, y_test = load_data("/mnistROT.npy", "/mnistROTLabel.npy", "/mnistROTTEST.npy", "/mnistROTLABELTEST.npy", "ROT_MNIST")
    fake_inputs = np.array(np.identity(batch_size * 10), dtype = np.float32)
    
    X_train = extend_image(X_train, 40)
    X_test = extend_image(X_test, 40)

    
    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')

    # The dimension would be (n, )
    vanilla_target_var = T.ivector('vanilla_targets')
    # The dimension would be (nRotation * n , )
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    
    network, weight_decay, network_transformed, side_network = build_cnn(input_var, batch_size)
    
    affine_network_size = 1000
    
    network_affine, _, network_transformed, side_network = build_cnn(input_var, affine_network_size)
    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    saved_weights = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_hinge.npy")

    # affine_matrix_array = np.array([ 1.,  0.,  0.,  0.,  1.,  0.], dtype = np.float32)
    # affine_matrix_matrix = np.array(np.random.normal(0, 3, (batch_size * 10 , )), dtype = np.float32)

    # degree_array = - np.arange(10) * np.pi * 2.0 + np.pi
    # affine_matrix_matrix = np.zeros((batch_size, 10))
    # affine_matrix_matrix[:,] = degree_array

    # affine_matrix_matrix = np.array(affine_matrix_matrix, dtype = np.float32).reshape(batch_size * 10, )
    
    affine_matrix_matrix = np.array(np.zeros((batch_size * 10,)), dtype = np.float32)
    
    affine_matrix_matrix_new_net = np.array(np.zeros((affine_network_size * 10,)), dtype = np.float32)

    # affine_matrix_matrix = np.clip(affine_matrix_matrix, -np.pi, np.pi)

    network_saved_weights = np.array([affine_matrix_matrix,] + [saved_weights[i] for i in range(saved_weights.shape[0])])
    
    affine_network_saved_weights = np.array([affine_matrix_matrix_new_net,] + [saved_weights[i] for i in range(saved_weights.shape[0])])

    lasagne.layers.set_all_param_values(network, network_saved_weights)
    
    lasagne.layers.set_all_param_values(network_affine, affine_network_saved_weights)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)
    predictions = lasagne.layers.get_output(network)
    
    predictions_new = lasagne.layers.get_output(network_affine)
    
    transformed_images = lasagne.layers.get_output(network_transformed)
    
    side_network_output = lasagne.layers.get_output(side_network)

    # The diVmension would be (nRotation * n, 10)
    one_hot_targets = T.extra_ops.to_one_hot(target_var, 10)

    predictions_affine = predictions_new[one_hot_targets.nonzero()].reshape((affine_network_size, 10))
    
    predictions_for_loss = predictions[one_hot_targets.nonzero()].reshape((batch_size, 10))

    predictions = predictions.reshape((batch_size, 10, 10))

    loss_affine_before = lasagne.objectives.squared_error(predictions_affine.clip(-20, 3), 3) * 2000 + lasagne.objectives.squared_error(predictions_affine.clip(3, 20), 20)
    loss_affine = loss_affine_before.mean()
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = one_vs_all_hinge_loss(final_rests, vanilla_target_var)
    loss = lasagne.objectives.multiclass_hinge_loss(predictions_for_loss, vanilla_target_var)
    loss = loss.mean() + weight_decay

    train_acc = T.mean(T.eq(T.argmax(predictions_for_loss, axis = 1), vanilla_target_var), dtype = theano.config.floatX)
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    params_newnet = lasagne.layers.get_all_params(network_affine, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    

    affine_params = params[0]
    model_params = params[1:]
    
    affine_params_newnet = params_newnet[0]
    model_params_newnet = params_newnet[1:]
    

    updates_affine = lasagne.updates.sgd(loss_affine, [affine_params_newnet], learning_rate = 10)
    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    test_prediction_newnet = lasagne.layers.get_output(network_affine, deterministic=True)
    
    test_prediction_affine = test_prediction_newnet[one_hot_targets.nonzero()].reshape((affine_network_size, 10))

    test_prediction = test_prediction[one_hot_targets.nonzero()].reshape((batch_size, 10))

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    loss_test_affine = lasagne.objectives.squared_error(test_prediction_affine.clip(-20, 3), 3) * 2000 + lasagne.objectives.squared_error(test_prediction_affine.clip(3, 20), 20)

    loss_test_affine = loss_test_affine.mean()

    update_affine_test = lasagne.updates.adagrad(loss_test_affine, [affine_params_newnet], learning_rate = 10)
    
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var, target_var], [loss,train_acc], updates=updates_model)
    
    train_affine_fn = theano.function([input_var, target_var], [loss_affine], updates=updates_affine)
    
    val_affine_fn = theano.function([input_var, target_var], [loss_test_affine], updates=update_affine_test)
    
    val_fn = theano.function([input_var, vanilla_target_var, target_var], test_acc)
    

    # Finally, launch the training loop.
    # We iterate over epochs:
    cached_affine_matrix = np.array(np.zeros((X_train.shape[0], 10,)), dtype = np.float32)
    cached_affine_matrix_test = np.array(np.zeros((X_test.shape[0], 10,)), dtype = np.float32)
    for epoch in range(num_epochs):
        
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
    

        print("Starting training...")
        train_err = 0
        train_acc_sum = 0
        train_batches = 0
        affine_train_batches = 0
        start_time = time.time()
        if epoch % 200 == 0:
            print(X_train.shape)
            for batch in iterate_minibatches(X_train, y_train, affine_network_size, shuffle=True):
                inputs, targets, index = batch
                affine_params_newnet.set_value(cached_affine_matrix[index].reshape(-1,))
                inputs = inputs.reshape(affine_network_size, 1, 40, 40)
                transformed_image_res = None
                fake_targets = np.zeros((affine_network_size, 10))
                fake_targets[:, ] = np.arange(10)
                fake_targets = np.array(fake_targets.reshape((affine_network_size * 10,)), dtype = np.int32)
                for i in range(100):
                    train_loss = train_affine_fn(inputs, fake_targets)
                weightsOfParams = lasagne.layers.get_all_param_values(network_affine)
                cached_affine_matrix[index] = weightsOfParams[0].reshape(-1, 10)
                affine_train_batches += 1
                print(affine_train_batches)

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
    np.save("../data/mnist_CNN_params_drop_out_Chi_2017_ROT_hinge_2000_em_final_2.npy", weightsOfParams)


if __name__ == '__main__':
    main()
