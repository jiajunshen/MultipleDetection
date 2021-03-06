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
from lasagne.layers import BatchNormLayer
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data
from lasagne.utils import as_theano_expression
from lasagne.regularization import regularize_layer_params_weighted, l2

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

def build_cnn(input_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 48, 48),
                                        input_var=input_var)

    network = BatchNormLayer(network)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    conv1 = Conv2DLayer(
            network, num_filters=5, filter_size=(5, 5),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())


    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    #network = MaxPool2DLayer(network, pool_size=(2, 2))
    
    #network = BatchNormLayer(network)

    """
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = Conv2DLayer(
            network, num_filters=10, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = BatchNormLayer(network)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    fc1   = lasagne.layers.DenseLayer(
            #lasagne.layers.dropout(network, p=.5),
            network,
            num_units=10,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )
    """

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2   = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(conv1, p=.5),
            #conv1,
            num_units=1,
            # nonlinearity=lasagne.nonlinearities.rectify,
            # nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity = lasagne.nonlinearities.identity
            )
    
    #weight_decay_layers = {fc1:0.0, fc2:0.00}
    weight_decay_layers = {conv1: 0.2}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc2, l2_penalty



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
    print("Loading data...")
    num_per_class = 100
    print("Using %d per class" % num_per_class) 
    
    #X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    
    X_train, y_train, X_test, y_test = load_data("/X_train_3_classes.npy", "/Y_train_3_classes.npy", "/X_test_3_classes.npy", "/Y_test_3_classes.npy")

    y_train[y_train == 2] = 0
    y_test[y_test == 2] = 0

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    # Create neural network model (depending on first command line parameter)

    network, weight_decay = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss_before = binary_hinge_loss(prediction, target_var)
    loss = loss_before.mean() + weight_decay
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.0001, momentum=0.9)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)
    #updates = lasagne.updates.adagrad(loss, params, learning_rate = 0.01)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = binary_hinge_loss(test_prediction,target_var)
    test_loss = test_loss.mean() + weight_decay
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq((test_prediction.reshape((-1,)) > 0.0), (target_var.reshape((-1,)))),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [loss,loss_before, prediction, test_acc], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction, (test_prediction.reshape((-1,)) > 0.0)])
    
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
            inputs, targets = batch
            targets = targets.reshape(100, 1)
            train_batch_err, train_batch_err_before, train_batch_prediction, train_batch_acc = train_fn(inputs, targets)
            train_acc += train_batch_acc
            train_err += train_batch_err
            train_batches += 1
            if (train_batches == 1 and epoch % 10 == 0):
                print("target label", targets[:20].reshape(-1,))
                print("loss before", train_batch_err_before[:20].reshape(-1,))
                print("train prediction", train_batch_prediction[:20].reshape(-1,))


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training acc:\t\t{:.6f}".format(train_acc / train_batches))

        if epoch % 10 == 0: 
            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 100, shuffle=True):
                inputs, targets = batch
                targets = targets.reshape(100, 1)
                err, acc, pred_before, pred = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
                if test_batches == 1:
                    print (pred[:5], targets[:5])
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))




if __name__ == '__main__':
    main()
