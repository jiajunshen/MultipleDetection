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
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data
from lasagne.layers import LocalResponseNormalization2DLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, InputLayer, DimshuffleLayer, BatchNormLayer
from lasagne.regularization import regularize_layer_params_weighted, l2


def build_teacher_cnn(input_var=None, matching_layer=2):

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
                               num_units=10,
                               nonlinearity=lasagne.nonlinearities.softmax,
                               W=lasagne.init.GlorotUniform(),
                               b=lasagne.init.Constant(0.0),
                               name="softmax")

    intermediate_layer = pool2

    return softmax_layer, intermediate_layer


def build_student_cnn(input_var=None, matching_layer=2):

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
                               num_units=10,
                               nonlinearity=lasagne.nonlinearities.softmax,
                               W=lasagne.init.GlorotUniform(),
                               b=lasagne.init.Constant(0.0),
                               name="softmax")

    intermediate_layer = pool2

    return softmax_layer, intermediate_layer

def iterate_minibatches_pair(inputs_teacher, inputs_student, targets, batchsize, shuffle=False):
    assert len(inputs_teacher) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs_teacher))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs_teacher) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs_teacher[excerpt], inputs_student[excerpt], targets[excerpt]


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


def main(model='mlp', num_epochs=1000):
    # Load the dataset
    print("Loading data...")

    X_teacher_train, y_teacher_train, _, _ = load_data("/X_train_plain.npy",
                                                 "/Y_train.npy",
                                                 "/X_small_train_plain.npy",
                                                 "/Y_small_train.npy",
                                                 resize=False,
                                                 standardize=True,
                                                 size = 32)
    X_student_train, y_student_train, X_student_test, y_student_test = \
        load_data("/X_train_texture.npy", "/Y_train.npy",
                  "/X_test_texture.npy", "/Y_test.npy",
                  resize=False, standardize=True, size=32)

    print(X_teacher_train.shape)
    print(X_student_train.shape)

    student_input_var = T.tensor4('inputs')
    teacher_input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    teacher_network, teacher_intermediate_layer = \
        build_teacher_cnn(teacher_input_var, 3)
    student_network, student_intermediate_layer = \
        build_student_cnn(student_input_var, 3)

    teacher_intermediate_layer_result = \
        lasagne.layers.get_output(teacher_intermediate_layer,
                                  deterministic=False)

    student_intermediate_layer_result = \
        lasagne.layers.get_output(student_intermediate_layer,
                                  deterministic=True)

    student_network_prediction = lasagne.layers.get_output(student_network)

    #loss = lasagne.objectives.categorical_crossentropy(student_network_prediction, target_var)
    #loss = loss.mean()

    loss = lasagne.objectives.squared_error(teacher_intermediate_layer_result,
                                            student_intermediate_layer_result)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(student_intermediate_layer,
                                           trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    #updates = []
    updates = lasagne.updates.adagrad(loss, params, learning_rate = 0.01)

    test_prediction = lasagne.layers.get_output(student_network,
                                                deterministic=True)

    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([teacher_input_var,
                                student_input_var], [loss, test_prediction], updates=updates)

    val_fn = theano.function([student_input_var, teacher_input_var, target_var],
                             [test_loss, test_acc, teacher_intermediate_layer_result,
                              student_intermediate_layer_result])

    # Loading the weights

    weightsOfIntermediateParams = \
        lasagne.layers.get_all_params(student_intermediate_layer)

    weightsOfTeacherNetwork = np.load("../data/plain_rotation_network_ShapeNet.npy")

    network_saved_weights = np.array([weightsOfIntermediateParams[i].eval() for i in range(8)] +
                                     [weightsOfTeacherNetwork[i] for i in range(8, weightsOfTeacherNetwork.shape[0])])


    lasagne.layers.set_all_param_values(student_network, network_saved_weights)
    lasagne.layers.set_all_param_values(teacher_network, weightsOfTeacherNetwork)
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches_pair(X_teacher_train, X_student_train,
                                         y_student_train, 100, shuffle=True):
            input_teacher, input_student, targets = batch
            err, prediction = train_fn(input_teacher, input_student)
            # print(prediction[0])
            # print(prediction[1])
            # time.sleep(2)
            train_err += err
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
            print(X_student_test.shape)
            print(X_student_train.shape)
            for batch in iterate_minibatches(X_student_test, y_student_test, 100, shuffle=False):
                inputs, targets = batch
                err, acc, tea_out, stu_out = val_fn(inputs, inputs, targets)
                #print(tea_out[0])
                #print(stu_out[0])
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

    # weightsOfParams = lasagne.layers.get_all_param_values(network)
    # np.save("../data/plain_rotation_network.npy", weightsOfParams)


if __name__ == '__main__':
    main()
