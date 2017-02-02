import os
os.environ['THEANO_FLAGS']='device=gpu2'

import numpy as np
np.random.seed(123)
import lasagne
import theano
import sys
import time
import theano.tensor as T
from lasagne.layers.dnn import Conv2DDNNLayer as conv
from lasagne.layers.dnn import MaxPool2DDNNLayer as pool
from dataPreparation import load_data

DIM = 42

def build_model(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, DIM, DIM),
                                     input_var = input_var)

    # Localization network
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()
    loc_l1 = pool(l_in, pool_size=(2, 2))
    loc_l2 = conv(
        loc_l1, num_filters=20, filter_size=(5, 5), W=lasagne.init.HeUniform())
    loc_l3 = pool(loc_l2, pool_size=(2, 2))
    loc_l4 = conv(loc_l3, num_filters=20, filter_size=(5, 5), W=lasagne.init.HeUniform())
    loc_l5 = pool(loc_l4, pool_size=(2, 2))
    loc_l6 = lasagne.layers.DenseLayer(
        loc_l5, num_units=20, W=lasagne.init.HeUniform('relu'))
    loc_out = lasagne.layers.DenseLayer(
        loc_l6, num_units=6, b=b, W=lasagne.init.Constant(0.0), 
        nonlinearity=lasagne.nonlinearities.identity)
    
    # Transformer network
    l_trans1 = lasagne.layers.TransformerLayer(l_in, loc_out, downsample_factor=1.0)
    print "Transformer network output shape: ", l_trans1.output_shape
    
    # Classification network
    class_l1 = conv(
        l_trans1,
        num_filters=32,
        filter_size=(9, 9),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    class_l2 = pool(class_l1, pool_size=(2, 2))
    class_l3 = conv(
        class_l2,
        num_filters=64,
        filter_size=(7, 7),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    class_l4 = pool(class_l3, pool_size=(2, 2))
    class_l5 = lasagne.layers.DenseLayer(
        #lasagne.layers.dropout(class_l4,p=.5),
        class_l4,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_out = lasagne.layers.DenseLayer(
        #lasagne.layers.dropout(class_l5,p=.5),
        class_l5,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
    )

    return l_out, l_trans1, loc_out

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
    margin_size = (size - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images

def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class)

    X_train, y_train, X_test, y_test = load_data("/mnistROT.npy", "/mnistROTLabel.npy", "/mnistROTTEST.npy", "/mnistROTLABELTEST.npy", "ROT_MNIST")
    # X_train, y_train, X_test, y_test = load_data("/X_train_limited_100.npy", "/Y_train_limited_100.npy", "/X_test.npy", "/Y_test.npy")
    
    """
    X_train_final = []
    y_train_final = []
    for i in range(10):
        X_train_class = X_train[y_train == i]
        permutated_index = np.arange(X_train_class.shape[0])
        X_train_final.append(X_train_class[permutated_index[:num_per_class]])
        y_train_final += [i] * num_per_class
    X_train = np.vstack(X_train_final)
    y_train = np.array(y_train_final, dtype = np.int32)
    """

    X_train = extend_image(X_train, DIM) / 255.0
    X_test = extend_image(X_test, DIM) / 255.0
    #X_train, y_train, X_test, y_test = load_data("/cluttered_train_x.npy", "/cluttered_train_y.npy", "/cluttered_test_x.npy", "/cluttered_test_y.npy", dataset = "MNIST_CLUTTER")
    _, _, X_test_rotated, y_test_rotated = load_data("/mnistROT.npy", "/mnistROTLabel.npy", "/mnistROTTEST.npy", "/mnistROTLABELTEST.npy", "ROT_MNIST")
    X_test_rotated = extend_image(X_test_rotated, DIM) / 255.0

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network, transformed_image, six_params = build_model(input_var)
    prediction = lasagne.layers.get_output(network, deterministic = False)
    prediction_eval = lasagne.layers.get_output(network, deterministic = True)
    transformed_image_eval = lasagne.layers.get_output(transformed_image, deterministic = True)
    six_params_eval = lasagne.layers.get_output(six_params, deterministic = True)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    eval_loss = lasagne.objectives.categorical_crossentropy(prediction_eval,
                                                            target_var)
    eval_loss = eval_loss.mean()
    eval_acc = T.mean(T.eq(T.argmax(prediction_eval, axis = 1), target_var),
                      dtype=theano.config.floatX)

    sh_lr = theano.shared(lasagne.utils.floatX(0.00001))
 
    params = lasagne.layers.get_all_params(network, trainable = True)
    updates = lasagne.updates.adam(loss, params, learning_rate = sh_lr)

    train_fn = theano.function([input_var, target_var], [loss,transformed_image_eval], updates = updates)
    eval_fn = theano.function([input_var, target_var], [eval_loss, eval_acc, transformed_image_eval, six_params_eval])

    print("Starting training...")

    for epoch in range(num_epochs):
        if (epoch + 1) % 200 == 0:
            new_lr = sh_lr.get_value() * 0.7
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 256, shuffle=True):
            inputs, targets = batch
            train_err, trans_img = train_fn(inputs, targets)
            train_err += train_err
            train_batches += 1
            if train_batches == 1:
                np.save("./unrotated_image_train.npy", trans_img)
                np.save("./original_image_train.npy", inputs)
        print(train_batches)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches)) 


        if epoch % 5 == 0:
            # After training, we compute and print the test error:
            print ("Start Evaluating")
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc, _, _ = eval_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

            print ("Start Evaluating")
            rotated_test_err = 0
            rotated_test_acc = 0
            rotated_test_batches = 0
            for batch in iterate_minibatches(X_test_rotated, y_test_rotated, 500, shuffle=False):
                inputs, targets = batch
                err, acc, trans_img, six_value = eval_fn(inputs, targets)
                rotated_test_err += err
                rotated_test_acc += acc
                rotated_test_batches += 1
                if rotated_test_batches == 1:
                    print(np.sum(six_value == 0))
                    print(six_value)
                    np.save("./unrotated_image.npy", trans_img)
                    np.save("./original_image.npy", inputs)
            print("Final results:")
            print("  rotated test loss:\t\t\t{:.6f}".format(rotated_test_err / rotated_test_batches))
            print("  rotated test accuracy:\t\t{:.2f} %".format(
                rotated_test_acc / rotated_test_batches * 100))

    weightsOfParams = lasagne.layers.get_all_param_values(network)

    np.save("../data/mnist_2017_spn.npy", weightsOfParams)



if __name__ == '__main__':
    main()

