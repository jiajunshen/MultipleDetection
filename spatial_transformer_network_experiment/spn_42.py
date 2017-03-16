import os
os.environ['THEANO_FLAGS']='device=gpu'

import matplotlib
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
from load_data_42 import load_data
from load_model_fc import build_model
conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer
NUM_EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.01
DIM = 42
NUM_CLASSES = 10
mnist_cluttered = "/phddata/jiajun/Research/mnist/rotated_mnist_42.npz"

def main():
    data = load_data(mnist_cluttered)
    model, l_transform, localize_output = build_model(DIM, DIM, NUM_CLASSES)
    model_params = lasagne.layers.get_all_params(model, trainable=True)

    X = T.tensor4()
    y = T.ivector()

    # training output
    output_train = lasagne.layers.get_output(model, X, deterministic=False)
    localize_output_value = lasagne.layers.get_output(localize_output, X, deterministic=True)

    # evaluation output. Also includes output of transform for plotting
    output_eval, transform_eval = lasagne.layers.get_output([model, l_transform], X, deterministic=True)

    sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
    cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
    #updates = lasagne.updates.adam(cost, model_params, learning_rate=sh_lr)
    updates = lasagne.updates.momentum(cost, model_params, learning_rate=sh_lr, momentum=0.9)

    train = theano.function([X, y], [cost, output_train, localize_output_value], updates=updates)
    eval = theano.function([X], [output_eval, transform_eval, localize_output_value])
    
    def train_epoch(X, y):
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
        costs = []
        correct = 0
        localized_output_list = []
        for i in range(num_batches):
            idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
            X_batch = X[idx]
            y_batch = y[idx]
            cost_batch, output_train, localize_output_batch = train(X_batch, y_batch)
            localized_output_list.append(localize_output_batch)
            costs += [cost_batch]
            preds = np.argmax(output_train, axis=-1)
            correct += np.sum(y_batch == preds)
        localized_output_list = np.vstack(localized_output_list)
        return np.mean(costs), correct / float(num_samples), localized_output_list


    def eval_epoch(X, y):
        output_eval, transform_eval, localized_output_list = eval(X)
        preds = np.argmax(output_eval, axis=-1)
        acc = np.mean(preds == y)
        return acc, transform_eval, localized_output_list


    valid_accs, train_accs, test_accs = [], [], []
    try:
        for n in range(NUM_EPOCHS):
            train_cost, train_acc, train_localized = train_epoch(data['X_train'], data['y_train'])
            valid_acc, valid_trainsform, valid_localized = eval_epoch(data['X_valid'], data['y_valid'])
            test_acc, test_transform, test_localized = eval_epoch(data['X_test'], data['y_test'])
            valid_accs += [valid_acc]
            test_accs += [test_acc]
            train_accs += [train_acc]

            if (n+1) % 20 == 0:
                new_lr = sh_lr.get_value() * 0.7
                print "New LR:", new_lr
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

            print "Epoch {0}: Train cost {1}, Train acc {2}, val acc {3}, test acc {4}".format(
                    n, train_cost, train_acc, valid_acc, test_acc)
            if n + 1 == NUM_EPOCHS:
                np.save("train_localized.npy", train_localized)
                np.save("test_localized.npy", test_localized)
    except KeyboardInterrupt:
        pass
    """
    plt.figure(figsize=(9,9))
    plt.plot(1-np.array(train_accs), label='Training Error')
    plt.plot(1-np.array(valid_accs), label='Validation Error')
    plt.legend(fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.savefig('trainingErrorPlot.pdf')
    """
    np.save("trainError.npy", [1 - np.array(train_accs), 1 - np.array(valid_accs)])

    """
    plt.figure(figsize=(7,14))
    for i in range(3):
        plt.subplot(321+i*2)
        plt.imshow(data['X_test'][i].reshape(DIM, DIM), cmap='gray', interpolation='none')
        if i == 0:
            plt.title('Original 60x60', fontsize=20)
        plt.axis('off')
        plt.subplot(322+i*2)
        plt.imshow(test_transform[i].reshape(DIM//3, DIM//3), cmap='gray', interpolation='none')
        if i == 0:
            plt.title('Transformed 20x20', fontsize=20)
        plt.axis('off')
    plt.savefig('transformed_image.pdf')
    """
    original_data = data['X_test'][:100].reshape(100, DIM, DIM)
    transformed_data = test_transform[:100].reshape(100, DIM, DIM)
    np.save("transformed_data.npy", transformed_data)
    np.save("original_data.npy", original_data)

    weightsOfParams = lasagne.layers.get_all_param_values(model)
    np.save("spn_network_weight.npy", weightsOfParams)


if __name__ == "__main__":
    main()
