# This file is a editted version of https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
# Use Lasagne for digit recognition using  MNIST dataset.
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
#from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data
from lasagne.layers import LocalResponseNormalization2DLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, InputLayer, DimshuffleLayer, BatchNormLayer
from lasagne.regularization import regularize_layer_params_weighted, l2



def build_cnn(input_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)

    #norm0 = BatchNormLayer(network)

    # conv1
    conv1 = Conv2DLayer(network, num_filters=64, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.0),
                        name="conv1")

    conv1a = Conv2DLayer(conv1, num_filters=64, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.0),
                        name="conv1a")


    pool1 = MaxPool2DLayer(conv1a, pool_size=(2, 2), stride=(2, 2), pad=0)

    #norm1 = BatchNormLayer(pool1)
    # pool1


    # conv2
    conv2 = Conv2DLayer(lasagne.layers.dropout(pool1, p = 0.5),
                        num_filters=128, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.1),
                        name='conv2')

    conv2a = Conv2DLayer(conv2,
                        num_filters=128, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.1),
                        name='conv2a')

    pool2 = MaxPool2DLayer(conv2a, pool_size=(2, 2), stride=(2, 2), pad=0)

    # norm2
    #norm2 = BatchNormLayer(pool2)

    # pool2


    conv3 = Conv2DLayer(lasagne.layers.dropout(pool2, p = 0.5),
                        num_filters=256, filter_size=(3,3),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.GlorotUniform(),
                        b=lasagne.init.Constant(0.1),
                        name='conv3')

    pool3 = MaxPool2DLayer(conv3, pool_size=(2, 2), stride=(2, 2), pad=0)

    #norm3 = BatchNormLayer(pool3)

    # fc1
    fc1 = DenseLayer(lasagne.layers.dropout(pool3, p = 0.5),
                     num_units=256,
                     nonlinearity=lasagne.nonlinearities.rectify,
                     W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1),
                     name="fc1")

    # fc3
    softmax_layer = DenseLayer(lasagne.layers.dropout(fc1, p = 0.5),
                               num_units=9,
                               nonlinearity=lasagne.nonlinearities.softmax,
                               W=lasagne.init.GlorotUniform(),
                               b=lasagne.init.Constant(0.0),
                               name="softmax")

    intermediate_layer = pool2

    return softmax_layer




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


def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")

    X_train, y_train, X_test, y_test = load_data(#"/imagenet_train.npy",
                                                 #"/imagenet_train_label.npy",
                                                 "/real_background_10_class_train.npy",
                                                 #"/white_background_10_class_train.npy",
                                                 "/10_class_train_label.npy",
                                                 #"/imagenet_test.npy",
                                                 #"/real_background_10_class_test.npy",
                                                 #"/10_class_test_label.npy",
                                                 "/imagenet_test_declutter.npy",
                                                 "/imagenet_test_label.npy",
                                                 resize=False,
                                                 standardize=True)
    train_num = 20000
    total_num_per_class = 20000
    X_train = np.vstack([X_train[i * total_num_per_class:i * total_num_per_class + train_num] for i in range(9)])
    y_train = np.concatenate([y_train[i * total_num_per_class : i * total_num_per_class + train_num] for i in range(9)])


    print(X_train.shape, X_test.shape)

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adagrad(loss, params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])

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
            cur_loss, cur_pre = train_fn(inputs, targets)
            train_batches == 0
            # print(cur_pre[:5])
            # print(cur_loss)
            train_err += cur_loss
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
            test_batches_0= 0
            test_batches_1 = 0
            test_batches_2 = 0
            test_batches_3= 0
            test_batches_4 = 0
            test_batches_5 = 0
            test_batches_6= 0
            test_batches_7 = 0
            test_batches_8 = 0
            test_err_0 = 0
            test_err_1 = 0
            test_err_2 = 0
            test_err_3 = 0
            test_err_4 = 0
            test_err_5 = 0
            test_err_6 = 0
            test_err_7 = 0
            test_err_8 = 0
            for batch in iterate_minibatches(X_test, y_test, 100, shuffle=False):
                inputs, targets = batch
                err, acc, pre = val_fn(inputs, targets)
                if test_batches == 0:
                    print(pre[:5])
                pre = np.argmax(pre, axis = 1)
                if np.sum(targets == 0) != 0:
                    test_err_0 += np.mean(pre[targets == 0] == 0)
                    test_batches_0 += 1
                if np.sum(targets == 1) != 0:
                    test_err_1 += np.mean(pre[targets == 1] == 1)
                    test_batches_1 += 1
                if np.sum(targets == 2) != 0:
                    test_err_2 += np.mean(pre[targets == 2] == 2)
                    test_batches_2 += 1
                if np.sum(targets == 3) != 0:
                    test_err_3 += np.mean(pre[targets == 3] == 3)
                    test_batches_3 += 1
                if np.sum(targets == 4) != 0:
                    test_err_4 += np.mean(pre[targets == 4] == 4)
                    test_batches_4 += 1
                if np.sum(targets == 5) != 0:
                    test_err_5 += np.mean(pre[targets == 5] == 5)
                    test_batches_5 += 1
                if np.sum(targets == 6) != 0:
                    test_err_6 += np.mean(pre[targets == 6] == 6)
                    test_batches_6 += 1
                if np.sum(targets == 7) != 0:
                    test_err_7 += np.mean(pre[targets == 7] == 7)
                    test_batches_7 += 1
                if np.sum(targets == 8) != 0:
                    test_err_8 += np.mean(pre[targets == 8] == 8)
                    test_batches_8 += 1
                

                test_err += err
                test_acc += acc
                test_batches += 1
            
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))
            print("  test accuracy 0:\t\t{:.2f} %".format(
                np.float(test_err_0) / np.float(test_batches_0) * 100))
            print("  test accuracy 1:\t\t{:.2f} %".format(
                np.float(test_err_1) / np.float(test_batches_1) * 100))
            print("  test accuracy 2:\t\t{:.2f} %".format(
                np.float(test_err_2) / np.float(test_batches_2) * 100))
            print("  test accuracy 3:\t\t{:.2f} %".format(
                np.float(test_err_3) / np.float(test_batches_3) * 100))
            print("  test accuracy 4:\t\t{:.2f} %".format(
                np.float(test_err_4) / np.float(test_batches_4) * 100))
            print("  test accuracy 5:\t\t{:.2f} %".format(
                np.float(test_err_5) / np.float(test_batches_5) * 100))
            print("  test accuracy 6:\t\t{:.2f} %".format(
                np.float(test_err_6) / np.float(test_batches_6) * 100))
            print("  test accuracy 7:\t\t{:.2f} %".format(
                np.float(test_err_7) / np.float(test_batches_7) * 100))
            print("  test accuracy 8:\t\t{:.2f} %".format(
                np.float(test_err_8) / np.float(test_batches_8) * 100))


    weightsOfParams = lasagne.layers.get_all_param_values(network)
    for i in range(len(weightsOfParams)):
        print(weightsOfParams[i].shape)
    #np.save("../data/plain_rotation_network_withBatchNorm_simple.npy", weightsOfParams)


if __name__ == '__main__':
    main()
