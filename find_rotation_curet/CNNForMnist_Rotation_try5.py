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
from  lasagne.layers import LocalResponseNormalization2DLayer as LRN


# Turn nw*h to n*w*h*nRotation
def rotateImage_batch(image, num_rotation=16):
    result_list = []
    for i in range(num_rotation):
        # angle = (90.0 / num_rotation) * i - 45
        angle = (360.0 / num_rotation) * i
        rotated_inputs = np.array([rotateImage(image[j], angle) for j in range(image.shape[0])], dtype = np.float32)
        result_list.append(rotated_inputs)
    return np.array(result_list, dtype = np.float32)

def rotateImage(image, angle):
    if len(image.shape) == 3:
        image = image[0]
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    return np.array(result[:, :], dtype = np.float32)



def build_cnn(input_var=None, batch_size = None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 227, 227),
                                        input_var=input_var)

    network = Conv2DLayer(
            network, num_filters=96, filter_size=(11, 11),
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
            num_units=61,
            nonlinearity=lasagne.nonlinearities.identity,
            )

    weight_decay_layers = {fc1:0.0, fc2:0.0}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc4, l2_penalty

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


def extend_images(images, dim = 277):
    extended_images_res = np.pad(images, ((0,0),(0,0),(13,14),(13,14)), mode="wrap")
    return extended_images_res


def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class) 
    print("Using all the training data") 
    
    ## Load Data##
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

    X_train = extend_images(X_train, 227)
    X_test = extend_images(X_test, 227)

    y_train = y_train
    y_test = y_test

    ## Define Batch Size ##
    batch_size = 50
 
    ## Define nRotation for exhaustive search ##
    nRotation = 16

    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)
    network, weight_decay = build_cnn(input_var, batch_size)
    
    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    saved_weights = np.load("../data/curet_test_hinge_epoch_400_2pool.npy")
    lasagne.layers.set_all_param_values(network, saved_weights)

    predictions = lasagne.layers.get_output(network)

    one_hot_targets = T.extra_ops.to_one_hot(vanilla_target_var, 61)

    rests = T.reshape(predictions, (nRotation, -1, 61))

    final_rests = T.max(rests, 0)
    
    rests = T.swapaxes(rests, 0, 2)
    rests = T.swapaxes(rests, 0, 1)
    rests = rests[one_hot_targets.nonzero()]
    rests = T.max(rests, axis = 1)

    final_rests = T.set_subtensor(final_rests[one_hot_targets.nonzero()], rests)

    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = one_vs_all_hinge_loss(final_rests, vanilla_target_var)
    loss = lasagne.objectives.multiclass_hinge_loss(final_rests, vanilla_target_var, 5)
    loss = loss.mean() + weight_decay
    # We could add some weight decay as well here, see lasagne.regularization.


    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adagrad(loss, params, learning_rate = 0.001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.reshape(test_prediction,(nRotation, -1, 61))
    test_prediction_res = test_prediction.max(axis = 0)

    final_test_prediction_res = test_prediction[0]
    test_prediction_process = T.swapaxes(test_prediction, 0, 2)
    test_prediction_process = T.swapaxes(test_prediction_process, 0, 1)
    test_prediction_process = test_prediction_process[one_hot_targets.nonzero()]
    test_prediction_process = T.max(test_prediction_process, axis = 1)

    final_test_prediction_res = T.set_subtensor(final_test_prediction_res[one_hot_targets.nonzero()], test_prediction_process)

    test_loss = lasagne.objectives.multiclass_hinge_loss(final_test_prediction_res, vanilla_target_var)
    test_loss = test_loss.mean() + weight_decay
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction_res, axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, vanilla_target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, vanilla_target_var], [test_loss, test_acc])



    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            inputs = inputs.reshape(batch_size, 227, 227)
            inputs = rotateImage_batch(inputs, nRotation).reshape(batch_size * nRotation, 1, 227, 227)
            duplicated_targets = np.array([targets for i in range(nRotation)]).reshape(batch_size * nRotation,)
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))




        if epoch % 10 == 0:
           # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                inputs, targets = batch
                inputs = inputs.reshape(batch_size, 227, 227)
                inputs = rotateImage_batch(inputs, nRotation).reshape(batch_size * nRotation, 1, 227, 227)
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets = batch
                inputs = inputs.reshape(batch_size, 227, 227)
                inputs = rotateImage_batch(inputs, nRotation).reshape(batch_size * nRotation, 1, 227, 227)
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

    weightsOfParams = lasagne.layers.get_all_param_values(network)
    np.save("../data/curet_justRotation_try5.npy", weightsOfParams)

if __name__ == '__main__':
    main()
