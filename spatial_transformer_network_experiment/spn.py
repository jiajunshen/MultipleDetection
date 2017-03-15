import os
os.environ['THEANO_FLAGS']='device=gpu'

import matplotlib
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
from load_data import load_data
from load_model import build_model
conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DIM = 60
NUM_CLASSES = 10
mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"

def main():
    data = load_data(mnist_cluttered)
    model, l_transform = build_model(DIM, DIM, NUM_CLASSES)
    model_params = lasagne.layers.get_all_params(model, trainable=True)

    X = T.tensor4()
    y = T.ivector()

    # training output
    output_train = lasagne.layers.get_output(model, X, deterministic=False)

    # evaluation output. Also includes output of transform for plotting
    output_eval, transform_eval = lasagne.layers.get_output([model, l_transform], X, deterministic=True)

    sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
    cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
    updates = lasagne.updates.adam(cost, model_params, learning_rate=sh_lr)

    train = theano.function([X, y], [cost, output_train], updates=updates)
    eval = theano.function([X], [output_eval, transform_eval])
    
    def train_epoch(X, y):
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
        costs = []
        correct = 0
        for i in range(num_batches):
            idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
            X_batch = X[idx]
            y_batch = y[idx]
            cost_batch, output_train = train(X_batch, y_batch)
            costs += [cost_batch]
            preds = np.argmax(output_train, axis=-1)
            correct += np.sum(y_batch == preds)

        return np.mean(costs), correct / float(num_samples)


    def eval_epoch(X, y):
        output_eval, transform_eval = eval(X)
        preds = np.argmax(output_eval, axis=-1)
        acc = np.mean(preds == y)
        return acc, transform_eval


    valid_accs, train_accs, test_accs = [], [], []
    try:
        for n in range(NUM_EPOCHS):
            train_cost, train_acc = train_epoch(data['X_train'], data['y_train'])
            valid_acc, valid_trainsform = eval_epoch(data['X_valid'], data['y_valid'])
            test_acc, test_transform = eval_epoch(data['X_test'], data['y_test'])
            valid_accs += [valid_acc]
            test_accs += [test_acc]
            train_accs += [train_acc]

            if (n+1) % 20 == 0:
                new_lr = sh_lr.get_value() * 0.7
                print "New LR:", new_lr
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

            print "Epoch {0}: Train cost {1}, Train acc {2}, val acc {3}, test acc {4}".format(
                    n, train_cost, train_acc, valid_acc, test_acc)
    except KeyboardInterrupt:
        pass

    plt.figure(figsize=(9,9))
    plt.plot(1-np.array(train_accs), label='Training Error')
    plt.plot(1-np.array(valid_accs), label='Validation Error')
    plt.legend(fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.savefig('trainingErrorPlot.pdf')

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

if __name__ == "__main__":
    main()
