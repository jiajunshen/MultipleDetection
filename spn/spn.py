%matplotlib inline
import os
os.environ['THEANO_FLAGS']='device=gpu2'

import matplotlib
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer

def build_model(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 40, 40),
                                     input_var = input_var)

    # Localization network
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()
    loc_l1 = pool(l_in, pool_size=(2, 2))
    loc_l2 = conv(
        loc_l1, num_filters=20, filter_size=(5, 5), W=ini)
    loc_l3 = pool(loc_l2, pool_size=(2, 2))
    loc_l4 = conv(loc_l3, num_filters=20, filter_size=(5, 5), W=ini)
    loc_l5 = lasagne.layers.DenseLayer(
        loc_l4, num_units=50, W=lasagne.init.HeUniform('relu'))
    loc_out = lasagne.layers.DenseLayer(
        loc_l5, num_units=6, b=b, W=lasagne.init.Constant(0.0), 
        nonlinearity=lasagne.nonlinearities.identity)
    
    # Transformer network
    l_trans1 = lasagne.layers.TransformerLayer(l_in, loc_out, downsample_factor=1.0)
    print "Transformer network output shape: ", l_trans1.output_shape
    
    # Classification network
    class_l1 = conv(
        l_trans1,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    class_l2 = pool(class_l1, pool_size=(2, 2))
    class_l3 = conv(
        class_l2,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    class_l4 = pool(class_l3, pool_size=(2, 2))
    class_l5 = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(class_l4,p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_out = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(class_l5,p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
    )

    return l_out, l_trans1

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

def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    num_per_class = 100
    print("Using %d per class" % num_per_class)

    X_train, y_train, X_test, y_test = load_data("/X_train_rotated.npy", "/Y_train_rotated.npy", "/X_test.npy", "/Y_test.npy")
    X_train_final = []
    y_train_final = []
    for i in range(10):
        X_train_class = X_train[y_train == i]
        permutated_index = np.arange(X_train_class.shape[0])
        X_train_final.append(X_train_class[permutated_index[:num_per_class]])
        y_train_final += [i] * num_per_class
    X_train = np.vstack(X_train_final)
    y_train = np.array(y_train_final, dtype = np.int32)

    X_train = extend_image(X_train, 40)
    X_test = extend_image(X_test, 40)
    #X_train, y_train, X_test, y_test = load_data("/cluttered_train_x.npy", "/cluttered_train_y.npy", "/cluttered_test_x.npy", "/cluttered_test_y.npy", dataset = "MNIST_CLUTTER")
    _, _, X_test_rotated, y_test_rotated = load_data("/X_train.npy", "/Y_train.npy", "/X_test_rotated.npy", "/Y_test_rotated.npy")
    X_test_rotated = extend_image(X_test_rotated, 40)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_model(input_var)
    prediction = lasagne.layers.get_output(network, deterministic = False)
    prediction_eval = lasagne.layers.get_output(network, deterministic = True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    eval_loss = lasagne.objectives.categorical_crossentropy(prediction_eval,
                                                            target_var)
    eval_loss = eval_loss.mean()
    eval_acc = T.mean(T.eq(T.argmax(prediction_eval, axis = 1), target_var),
                      dtype=theano.config.floatX)
    
    params = lasagne.layers.get_all_params(network, trainable = True)
    updates = lasagne.updates.adagrad(loss, params, learning_rate = 0.01)

    train_fn = theano.function([input_var, target_var], loss, updates = updates)
    eval_fn = theano.function([input_var, target_var], [eval_loss, eval_acc])

    print("Starting training...")

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

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
                err, acc = val_fn(inputs, targets)
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
                err, acc = val_fn(inputs, targets)
                rotated_test_err += err
                rotated_test_acc += acc
                rotated_test_batches += 1
            print("Final results:")
            print("  rotated test loss:\t\t\t{:.6f}".format(rotated_test_err / rotated_test_batches))
            print("  rotated test accuracy:\t\t{:.2f} %".format(
                rotated_test_acc / rotated_test_batches * 100))

    weightsOfParams = lasagne.layers.get_all_param_values(network)

    np.save("../data/mnist_2017_spn.npy", weightsOfParams)



if __name__ == '__main__':
    main()

