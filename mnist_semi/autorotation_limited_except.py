# This file is a editted version of https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
# Use Lasagne for digit recognition using  MNIST dataset.
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer

def build_cnn(input_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 40, 40),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    #network_middle_output = lasagne.layers.ReshapeLayer(network, shape = (([0], 41472)))
    
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
    
    
    network_middle_output = lasagne.layers.ReshapeLayer(network, shape = (([0], 1568)))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )
    

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=10,
            )
    

    network = lasagne.layers.NonlinearityLayer(
            network,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network, network_middle_output

# build up the old network.
original_input_var = T.tensor4('original_inputs')
target_var = T.ivector('targets')
original_network, original_network_middle = build_cnn(original_input_var)
all_weights = np.load("../data/mnist_Chi_dec_100_test.npy")
# all_weights = np.load("./a.npy")
# all_weights = np.load("../data/mnist_CNN_params_drop_out_semi_Chi_Nov28_rot.npy")
lasagne.layers.set_all_param_values(original_network, all_weights)
original_network_middle_output = lasagne.layers.get_output(original_network_middle, original_input_var, deterministic = True)


def build_rotation_cnn(input_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 40, 40),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.Uniform(6.0/64))
    
    #network_middle_output = lasagne.layers.ReshapeLayer(network, shape = (([0], 41472)))
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            # W = all_weights[2],
            # b = all_weights[3],
            W = lasagne.init.Uniform(6.0/64)
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )


    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    
    network_middle_output = lasagne.layers.ReshapeLayer(network, shape = (([0], 1568)))
    #network = Conv2DLayer(
    #        network, num_filters=32, filter_size=(1, 1),
    #        nonlinearity=lasagne.nonlinearities.rectify,
    #        W = lasagne.init.GlorotUniform()
    #        #nonlinearity=lasagne.nonlinearities.sigmoid
    #        )
    #network = Conv2DLayer(
    #        network, num_filters=32, filter_size=(1, 1),
    #        nonlinearity=lasagne.nonlinearities.rectify,
    #        W = lasagne.init.GlorotUniform()
    #        #nonlinearity=lasagne.nonlinearities.sigmoid
    #        )
    
    #network_middle_output = lasagne.layers.NonlinearityLayer(network_middle_output, nonlinearity = lasagne.nonlinearities.sigmoid)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            W = all_weights[4], 
            b = all_weights[5],
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )
    

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            W = all_weights[6],
            b = all_weights[7],
            num_units=10,
            )
    
    
    
    network = lasagne.layers.NonlinearityLayer(
              network,
              nonlinearity=lasagne.nonlinearities.softmax)

    return network, network_middle_output

# Set up the rotated network

rotated_input_var = T.tensor4('rotated_inputs')
rotated_network, rotated_network_middle = build_rotation_cnn(rotated_input_var)
# previous_trained = np.load("../data/mnist_rotated_network_semi_Chi_Nov28.npy")

# lasagne.layers.set_all_param_values(rotated_network, previous_trained)

rotated_network_training_param = lasagne.layers.get_all_params(rotated_network_middle)
rotated_network_middle_output = lasagne.layers.get_output(rotated_network_middle, rotated_input_var, deterministic = False)
rotated_network_output = lasagne.layers.get_output(rotated_network, rotated_input_var, deterministic = True)

# build up the old network for the validation purpose
rotated_original_input_var = T.tensor4('rotated_inputs')
rotated_original_network, rotated_original_network_mid = build_cnn(rotated_original_input_var)
lasagne.layers.set_all_param_values(rotated_original_network, all_weights)
rotated_original_network_output = lasagne.layers.get_output(rotated_original_network, rotated_original_input_var, deterministic=True)

original_network_prediction_acc = T.mean(T.eq(T.argmax(rotated_original_network_output, axis = 1), target_var), dtype = theano.config.floatX)

rotated_network_prediction_acc = T.mean(T.eq(T.argmax(rotated_network_output, axis = 1), target_var), dtype = theano.config.floatX)

# Define loss function
L = T.mean(lasagne.objectives.squared_error(original_network_middle_output, rotated_network_middle_output), axis = 1)
cost = T.mean(L)

# updates = lasagne.updates.nesterov_momentum(cost, rotated_network_training_param, learning_rate = 0.001, momentum = 0.95)
updates = lasagne.updates.adagrad(cost, rotated_network_training_param, learning_rate = 0.01)

train_fn = theano.function(inputs = [original_input_var, rotated_input_var, target_var], 
                           outputs = [cost, rotated_network_prediction_acc], updates = updates)

val_fn = theano.function(inputs = [original_input_var, rotated_input_var, target_var], outputs = [cost, rotated_network_prediction_acc])
val_fn_original = theano.function(inputs = [rotated_original_input_var, target_var], outputs = original_network_prediction_acc)

import cv2
import numpy as np

def rotateImage(image, angle):
  if len(image.shape) == 3:
        image = image[0]
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return np.array(result[np.newaxis, :, :], dtype = np.float32)

def extend_image(inputs, size = 40):
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    margin_size = (40 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images

from CNNForMnist import build_cnn, load_data

num_train = 100
X_train, y_train, X_test, y_test = load_data("/X_train_limited_%d.npy" %num_train, "/Y_train_limited_%d.npy" %num_train, "/X_test.npy", "/Y_test.npy")
X_train = extend_image(X_train, 40)
#X_train = X_train[y_train == 7]
#y_train = y_train[y_train == 7]
X_test = extend_image(X_test, 40)
#X_test = X_test[y_test == 7]
#y_test = y_test[y_test == 7]

_, _, X_test_rotated, y_test_rotated = load_data("/X_train.npy", "/Y_train.npy", "/X_test_rotated.npy", "/Y_test_rotated.npy")

X_test_rotated = X_test_rotated[y_test_rotated == 8]
y_test_rotated = y_test_rotated[y_test_rotated == 8]

X_test_rotated = extend_image(X_test_rotated, 40)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

train_batches = X_train.shape[0] // 100
test_batches = X_test.shape[0] // 100
rotated_test_batches = X_test_rotated.shape[0] // 500
num_epochs = 2001
for epoch in range(num_epochs):
    if epoch != 0:
        accuracy = 0
        total_cost = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle = True):
            inputs, targets = batch
            angles_1 = list(np.random.randint(low = -20, high = -5, size = 50))
            angles_2 = list(np.random.randint(low = -5, high = 20, size = 50))
            angles = np.array(angles_1 + angles_2)
            np.random.shuffle(angles)
            angles[targets == 8] = 0
            rotated_inputs = np.array([rotateImage(inputs[i], angles[i]) for i in range(100)], dtype = np.float32)
            current_cost, current_accuracy = train_fn(inputs, rotated_inputs, targets)
            accuracy += current_accuracy
            total_cost += current_cost
        if epoch % 50 == 0:         
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                 epoch + 1, num_epochs, time.time() - start_time))
            print("training cost:\t\t{:.6f}".format(total_cost / train_batches))
            print("training accuracy:\t\t{:.6f}".format(accuracy / train_batches))
    
    if epoch % 100 == 0:
        print("Start Evaluating")
        test_accuracy = 0
        test_accuracy_on_original = 0
        original_test_accuracy = 0
        test_cost = 0
        for batch in iterate_minibatches(X_test, y_test, 100, shuffle = False):
            inputs, targets = batch
            angles_1 = list(np.random.randint(low = -20, high = -5, size = 50))
            angles_2 = list(np.random.randint(low = 5, high = 20, size = 50))
            angles = np.array(angles_1 + angles_2)
            np.random.shuffle(angles)
            
            rotated_inputs = np.array([rotateImage(inputs[i], angles[i]) for i in range(100)], dtype = np.float32)            
            current_original_accuracy = val_fn_original(rotated_inputs, targets)
            current_cost, current_accuracy = val_fn(inputs, rotated_inputs, targets)
            _, current_accuracy_on_original = val_fn(inputs, inputs, targets)
            test_accuracy += current_accuracy
            original_test_accuracy += current_original_accuracy
            test_accuracy_on_original += current_accuracy_on_original
            test_cost += current_cost

        print("Test Results:  ")
        print("approximate test cost:\t\t{:.6f}".format(test_cost / test_batches))
        print("approximate test accuracy:\t\t{:.6f}".format(test_accuracy / test_batches))
        print("test accuracy on original test dataset:\t\t{:.6f}".format(test_accuracy_on_original / test_batches))
        print("approximate Original Test Accuracy:\t\t{:.6f}".format(original_test_accuracy / test_batches))
        

        rotated_test_accuracy = 0
        original_rotated_test_accuracy = 0
        for batch in iterate_minibatches(X_test_rotated, y_test_rotated, 500, shuffle = False):
            inputs, targets = batch
            
            current_original_accuracy = val_fn_original(inputs, targets)
            _, current_accuracy = val_fn(inputs, inputs, targets)
            rotated_test_accuracy += current_accuracy
            original_rotated_test_accuracy += current_original_accuracy
        print("Rotated Test Results:  ")
        print("rotated test accuracy:\t\t{:.6f}".format(rotated_test_accuracy / rotated_test_batches))
        print("Original rotated Test Accuracy:\t\t{:.6f}".format(original_rotated_test_accuracy / rotated_test_batches))


rotatedNetworkParams = lasagne.layers.get_all_param_values(rotated_network)
np.save("../data/mnist_rotated_network_semi_Chi_Nov28_new.npy", rotatedNetworkParams)
