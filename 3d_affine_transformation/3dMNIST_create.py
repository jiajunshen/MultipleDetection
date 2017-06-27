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
from collections import OrderedDict
from affineTransformation3D import AffineTransformation3DLayer
from dataPreparation import load_data
import h5py
import amitgroup as ag
import amitgroup.plot as gr
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def build_cnn(input_var=None, batch_size = None):
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 40, 40, 40),
                                        input_var=input_var)
    network = AffineTransformation3DLayer(network, batch_size)

    return network


X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

X_train_2d = X_train.reshape(X_train.shape[0], 1, 28, 28, 1)
X_train_3d = np.zeros((X_train.shape[0], 1, 40, 40, 40))

X_train_3d[:, :, 6:34, 6:34, 17:23] = X_train_2d
X_train_3d = np.array(X_train_3d, dtype = np.float32)

X_test_2d = X_test.reshape(X_test.shape[0], 1, 28, 28, 1)
X_test_3d = np.zeros((X_test.shape[0], 1, 40, 40, 40))

X_test_3d[:, :, 6:34, 6:34, 17:23] = X_test_2d
X_test_3d = np.array(X_test_3d, dtype = np.float32)

input_var = T.tensor5('inputs')
network = build_cnn(input_var, 100)
network_output = lasagne.layers.get_output(network)
get_rotated = theano.function([input_var], network_output)


# Start Creating training 3d MNIST Data
image_result_list = []
for i in range(X_train_3d.shape[0] // 100):
    x_degree = np.random.randint(-45, 45, 100).reshape(100,)
    y_degree = np.random.randint(-45, 45, 100).reshape(100,)
    z_degree = np.random.randint(-45, 45, 100).reshape(100,)
    lasagne.layers.set_all_param_values(network, [np.array(x_degree,dtype=np.float32),
                                                  np.array(y_degree,dtype=np.float32),
                                                  np.array(z_degree,dtype=np.float32)])
    image_result = get_rotated(X_train_3d[100 * i : 100 * (i+1)])
    image_result_list.append(image_result)

train_image_result = np.vstack(image_result_list)
print(train_image_result.shape)
#train_data_file = h5py.File("/hdd/Documents/Data/3D-MNIST/X_train.hdf5", "w")
#train_data_file.create_dataset("train_data", data = train_image_result)
#train_data_file.create_dataset("train_label", data = y_train)
np.save("/hdd/Documents/Data/3D-MNIST/X_train.npy", train_image_result)
np.save("/hdd/Documents/Data/3D-MNIST/Y_train.npy", y_train)

# Start Creating testing 3d MNIST Data
image_result_list = []
for i in range(X_test_3d.shape[0] // 100):
    x_degree = np.random.randint(-45, 45, 100).reshape(100,)
    y_degree = np.random.randint(-45, 45, 100).reshape(100,)
    z_degree = np.random.randint(-45, 45, 100).reshape(100,)
    lasagne.layers.set_all_param_values(network, [np.array(x_degree,dtype=np.float32),
                                                  np.array(y_degree,dtype=np.float32),
                                                  np.array(z_degree,dtype=np.float32)])
    image_result = get_rotated(X_test_3d[100 * i : 100 * (i+1)])
    image_result_list.append(image_result)

test_image_result = np.vstack(image_result_list)
print(test_image_result.shape)
#test_data_file = h5py.File("/hdd/Documents/Data/3D-MNIST/X_test.hdf5", "w")
#test_data_file.create_dataset("test_data", data = test_image_result)
#test_data_file.create_dataset("test_data_label", data = y_test)
np.save("/hdd/Documents/Data/3D-MNIST/X_test.npy", test_image_result)
np.save("/hdd/Documents/Data/3D-MNIST/Y_test.npy", y_test)





PLOT = False
image_result = train_image_result
if PLOT:
    x_list = []
    y_list = []
    z_list = []
    dx, dy, dz = np.meshgrid(np.linspace(0, 39, 40), np.linspace(0, 39, 40), np.linspace(0, 39, 40))
    dx = np.array(dx, dtype=np.int32).flatten()
    dy = np.array(dy, dtype=np.int32).flatten()
    dz = np.array(dz, dtype=np.int32).flatten()
    for i in range(40 * 40 * 40):
        if X_train_3d[0, 0, dx[i], dy[i], dz[i]] > 0:
            x_list.append(dx[i])
            y_list.append(dy[i])
            z_list.append(dz[i])

    # Define a list of colors to plot atoms

    colors = ['r']


    ax = plt.subplot(111, projection='3d')

    # Plot scatter of points
    ax.scatter3D(x_list, y_list, z_list, c=colors)
    plt.show()



    x_list = []
    y_list = []
    z_list = []
    dx, dy, dz = np.meshgrid(np.linspace(0, 39, 40), np.linspace(0, 39, 40), np.linspace(0, 39, 40))
    dx = np.array(dx, dtype=np.int32).flatten()
    dy = np.array(dy, dtype=np.int32).flatten()
    dz = np.array(dz, dtype=np.int32).flatten()
    for i in range(40 * 40 * 40):
        if image_result[0, 0, dx[i], dy[i], dz[i]] > 0:
            x_list.append(dx[i])
            y_list.append(dy[i])
            z_list.append(dz[i])

    # Define a list of colors to plot atoms

    colors = ['r']


    ax = plt.subplot(111, projection='3d')

    # Plot scatter of points
    ax.scatter3D(x_list, y_list, z_list, c=colors)
    plt.show()
