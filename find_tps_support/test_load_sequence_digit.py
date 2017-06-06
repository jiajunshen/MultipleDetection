from dataPreparation import load_data_digit_clutter
import numpy as np
import os

a = load_data_digit_clutter("/X_train_limited_100.npy", "/Y_train_limited_100.npy", "/X_test.npy", "/Y_test.npy")

MNIST_PATH = os.environ['MNIST']
np.save(MNIST_PATH + "/X_test_sequence.npy", a[2])
np.save(MNIST_PATH + "/Y_test_sequence.npy", a[3])
