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


def build_cnn(input_var=None, constant_var = None, batch_size = 100):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(batch_size, 1, 40, 40),
                                        input_var=input_var)

    repeatInput = Repeat(network, 10)

    network = lasagne.layers.ReshapeLayer(repeatInput, (-1, 1, 40, 40))

    side_network = lasagne.layers.InputLayer(shape=(batch_size * 10, batch_size * 10),
                                             input_var=constant_var)

    side_network = lasagne.layers.DenseLayer(side_network,
                                             num_units=6, b = None)

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
            #network,
            num_units=10,
            )

    fc2_sigmoid = lasagne.layers.NonlinearityLayer(fc2, nonlinearity = lasagne.nonlinearities.sigmoid)

    # fc2 = lasagne.layers.ReshapeLayer(fc2, (-1, 10, 10))

    network_transformed = lasagne.layers.ReshapeLayer(network_transformed, (-1, 10, 40, 40))
    
    weight_decay_layers = {fc1:0.0, fc2:0.002}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc2, fc2_sigmoid, l2_penalty, network_transformed



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
    
    batch_size = 50 
    #X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test_rotated.npy", "/Y_test_rotated.npy")
    X_train, y_train, X_test, y_test = load_data("/mnistROT.npy", "/mnistROTLabel.npy", "/mnistROTTEST.npy", "/mnistROTLABELTEST.npy", "ROT_MNIST")
    fake_inputs = np.array(np.identity(batch_size * 10), dtype = np.float32)
    
    X_train = extend_image(X_train, 40)
    X_test = extend_image(X_test, 40)

    
    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')
    fake_input_var = T.matrix('fake_input')

    # The dimension would be (n, )
    vanilla_target_var = T.ivector('vanilla_targets')
    # The dimension would be (nRotation * n , )
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    
    network, network_sigmoid, weight_decay, network_transformed = build_cnn(input_var, fake_input_var, batch_size)
    
    # saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    saved_weights = np.load("../data/mnist_CNN_params_drop_out_Chi_2017_hinge.npy")

    # affine_matrix_array = np.array([ 1.,  0.,  0.,  0.,  1.,  0.], dtype = np.float32)
    affine_matrix_array = np.array([ 1.,  0.,  0.,  0.,  1.,  0.], dtype = np.float32)
    affine_matrix_matrix = np.zeros((batch_size * 10 , 6), dtype = np.float32)
    affine_matrix_matrix[:,] = affine_matrix_array.flatten()

    saved_weights = np.array([affine_matrix_matrix,] + [saved_weights[i] for i in range(saved_weights.shape[0])])
    lasagne.layers.set_all_param_values(network, saved_weights)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)
    predictions = lasagne.layers.get_output(network)
    
    predictions_sigmoid = lasagne.layers.get_output(network_sigmoid)
    
    transformed_images = lasagne.layers.get_output(network_transformed)

    # The diVmension would be (nRotation * n, 10)
    one_hot_targets = T.extra_ops.to_one_hot(target_var, 10)

    predictions_affine = predictions_sigmoid[one_hot_targets.nonzero()].reshape((batch_size, 10))
    
    predictions_for_loss = predictions[one_hot_targets.nonzero()].reshape((batch_size, 10))

    predictions = predictions.reshape((batch_size, 10, 10))

    loss_affine = lasagne.objectives.squared_error(predictions_affine, 1)
    loss_affine = loss_affine.mean()
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = one_vs_all_hinge_loss(final_rests, vanilla_target_var)
    loss = lasagne.objectives.multiclass_hinge_loss(predictions_for_loss, vanilla_target_var)
    loss = loss.mean() + weight_decay
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    

    affine_params = params[0]
    model_params = params[1:]
    

    updates_affine = lasagne.updates.adagrad(loss_affine, [affine_params], learning_rate = 0.1)
    updates_model = lasagne.updates.adagrad(loss, model_params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, vanilla_target_var)
    test_loss = test_loss.mean() + weight_decay
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:

    train_model_fn = theano.function([input_var, fake_input_var, vanilla_target_var, target_var], loss, updates=updates_model)
    
    train_affine_fn = theano.function([input_var, fake_input_var, target_var], [loss_affine, transformed_images, predictions], updates=updates_affine)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, fake_input_var, vanilla_target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            affine_params.set_value(affine_matrix_matrix)
            inputs, targets = batch
            inputs = inputs.reshape(batch_size, 1, 40, 40)
            transformed_image_res = None
            fake_targets = np.zeros((batch_size, 10))
            fake_targets[:, ] = np.arange(10)
            fake_targets = np.array(fake_targets.reshape((batch_size * 10,)), dtype = np.int32)
            for i in range(1000):
                train_loss, transformed_image_res, predictions_res = train_affine_fn(inputs, fake_inputs, fake_targets)
                print(train_loss)
                weightsOfParams = lasagne.layers.get_all_param_values(network)
                print(weightsOfParams[0][0])
            np.save("./transformed_images.npy", transformed_image_res)
            np.save("./original_images.npy", inputs)
            print (np.mean(transformed_image_res - inputs))
            train_err += train_fn(inputs, fake_inputs, targets, fake_targets)
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


        """
        if epoch % 50 == : 
           # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                inputs = inputs.reshape(500, 40, 40)
                inputs = rotateImage_batch(inputs, nRotation).reshape(500 * nRotation, 1, 40, 40)
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
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=False):
                inputs, targets = batch
                inputs = inputs.reshape(500, 40, 40)
                inputs = rotateImage_batch(inputs, nRotation).reshape(500 * nRotation, 1, 40, 40)
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
            """
            # Optionally, you could now dump the network weights to a file like this:
            # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
            #
            # And load them again later on like this:
            # with np.load('model.npz') as f:
            #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            # lasagne.layers.set_all_param_values(network, param_values)
    weightsOfParams = lasagne.layers.get_all_param_values(network)
    #np.save("../data/mnist_clutter_CNN_params_sigmoid.npy", weightsOfParams)
    #np.save("../data/mnist_CNN_params_sigmoid.npy", weightsOfParams)
    #np.save("../data/mnist_CNN_params.npy", weightsOfParams)
    #np.save("../data/mnist_CNN_params_drop_out_semi_Chi_Dec7.npy", weightsOfParams)
    np.save("../data/mnist_CNN_params_drop_out_Chi_2017_ROT_hinge_2000_em_test.npy", weightsOfParams)
    #np.save("../data/mnist_CNN_params_For_No_Bias_experiment_out.npy", weightsOfParams)



if __name__ == '__main__':
    main()