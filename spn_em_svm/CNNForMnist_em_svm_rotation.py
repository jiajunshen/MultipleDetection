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
from CNNForMnist_Rotation_Net import rotateImage_batch 
from CNNForMnist_Rotation_Net import rotateImage
from collections import OrderedDict
# from build_cnn import build_cnn
from build_cnn_only_rotation import build_cnn


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
    
    ## Load Data##
    X_train, y_train, X_test, y_test = load_data("/mnistROT.npy", "/mnistROTLabel.npy", "/mnistROTTEST.npy", "/mnistROTLABELTEST.npy", "ROT_MNIST")
    # X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

    
    X_train = extend_image(X_train, 40)
    X_test_all = extend_image(X_test, 40)
    X_test = extend_image(X_test, 40)

    y_train = y_train
    y_test_all = y_test[:]
    y_test = y_test


    ## Define Batch Size ##
    batch_size = 100
 
    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    vanilla_target_var = T.ivector('vanilla_targets')

    # Create neural network model (depending on first command line parameter)
    network, weight_decay, network_transformed, _ = build_cnn(input_var, batch_size)
    
    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    # saved_weights = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_hinge.npy")

    #initial_weights = lasagne.layers.get_all_param_values(network_transformed)

    #lasagne.layers.set_all_param_values(network, [initial_weights[i] for i in range(8)] + [saved_weights[i] for i in range(8)])
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)

    train_prediction = lasagne.layers.get_output(network)

    transformed_images = lasagne.layers.get_output(network_transformed, deterministic = True)
   
    loss = lasagne.objectives.multiclass_hinge_loss(train_prediction, vanilla_target_var)
    # loss = lasagne.objectives.categorical_crossentropy(train_prediction, vanilla_target_var)
    loss = loss.mean()
    # loss = loss.mean() + weight_decay

    # This is to use the rotation that the "gradient descent" think will give the highest score for each of the classes
    train_acc_1 = T.mean(T.eq(T.argmax(train_prediction, axis = 1), vanilla_target_var), dtype = theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    model_params = params
    # affine_params = params[:80]
    # model_params = params[:80]
    # updates_affine = lasagne.updates.sgd(loss_affine, affine_params, learning_rate = 0.01)
    
    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)
    # updates_model = lasagne.updates.momentum(loss, model_params, learning_rate = 0.001, momentum = 0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, vanilla_target_var], [loss, train_acc_1, transformed_images], updates=updates_model)
    
    # train_affine_fn = theano.function([input_var], [loss_affine, loss_affine_before], updates=updates_affine)

    val_fn = theano.function([input_var, vanilla_target_var], test_acc)
    

    # Finally, launch the training loop.
    # We iterate over epochs:

    for epoch in range(num_epochs):
        start_time = time.time()
        if epoch % 100 == 0 or epoch + 1 == num_epochs:
            print ("Start Evaluating...")
            test_acc = 0
            test_batches = 0
            # Find best rotation
            if epoch + 1 == num_epochs:
                X_test = X_test_all
                y_test = y_test_all

            for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
                inputs, targets, index = batch
                inputs = inputs.reshape(batch_size, 1, 40, 40)
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

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets, index = batch
            inputs = inputs.reshape(batch_size, 1, 40, 40)
            train_loss_value, train_acc_value_1, trained_image = train_model_fn(inputs, targets)
            train_err += train_loss_value
            train_acc_sum_1 += train_acc_value_1
            train_batches += 1
            if train_batches == 1:
                np.save("original_image.npy", inputs)
                np.save("transformed_image.npy", trained_image)
            #sys.exit()
        
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc 1:\t\t{:.6f}".format(train_acc_sum_1 / train_batches))
        
        #if epoch % 100 == 0:
        #    weightsOfParams = lasagne.layers.get_all_param_values(network)
        #    np.save("../data/mnist_CNN_params_drop_out_Chi_2017_ROT_hinge_2000_em_new_batch_run_epoch_%d.npy" %epoch, weightsOfParams)

        
    #weightsOfParams = lasagne.layers.get_all_param_values(network)
    #np.save("../data/mnist_CNN_params_drop_out_Chi_2017_ROT_hinge_2000_em_new_2.npy", weightsOfParams)


if __name__ == '__main__':
    main()
