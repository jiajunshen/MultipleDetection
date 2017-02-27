import os
import numpy as np
import sys
import time
import lasagne

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    #pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    #x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_test = x[50000:, :, :, :]
    Y_test = y[50000:]

    X_train_list = []
    Y_train_list = []
    for i in range(10):
        X_train_list.append(X_train[Y_train == i][:1000])
        Y_train_list.append(np.ones(1000) * i)

    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    pixel_mean = np.mean(X_train, axis = 0)
    X_train -= pixel_mean

    index = np.arange(10000)
    np.random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    #X_train_flip = X_train[:,:,:,::-1]
    #Y_train_flip = Y_train
    #X_train = np.concatenate((X_train,X_train_flip),axis=0)
    #Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test -= pixel_mean

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)
