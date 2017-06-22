# The model architecture is following: CNN with Anisotropic Probing kernels
# https://github.com/charlesq34/3dcnn.torch/blob/master/torch_models/aniprobing.lua
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data


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


def build_cnn(input_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 32, 32, 32),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
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
    network = NIN_block(network, 3, (128, 128, 40))
    network = MaxPool2DLayer(network, pool_size=(8, 8), stride=(1, 1))

    network = lasagne.layers.DenseLayer(
              network,
              num_units=40,
              nonlinearity=lasagne.nonlinearities.identity)
    """
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    #network = MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

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
            # nonlinearity=lasagne.nonlinearities.rectify,
            # nonlinearity=lasagne.nonlinearities.sigmoid
            )
    """

    return network


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


def main(model='mlp', num_epochs=3000):
    # Load the dataset
    print("Loading ModelNet40 data...")

    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)

    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = lasagne.objectives.multiclass_hinge_loss(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # params_value = lasagne.layers.get_all_param_values(network)
    # for param in params_value:
    #       print param.shape
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=0.01)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction,
                                                         target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    weightsOfParams = np.load("../data/NIN_ModelNet40_softmax_epoch1600.npy")
    lasagne.layers.set_all_param_values(network, weightsOfParams)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
            inputs, targets = batch
            train_batch_err, train_batch_prediction = train_fn(inputs, targets)
            train_err += train_batch_err
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        if epoch % 5 == 0:
            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
        if epoch % 100 == 0 and epoch != 0:
            weightsOfParams = lasagne.layers.get_all_param_values(network)
            np.save("../data/NIN_ModelNet40_hinge_epoch%d.npy" % epoch, weightsOfParams)



if __name__ == '__main__':
    main()
