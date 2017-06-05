from dataPreparation import load_data_digit_clutter
import numpy as np

a = load_data_digit_clutter("/X_train_limited_100.npy", "/Y_train_limited_100.npy", "/X_test.npy", "/Y_test.npy")
np.save("saveresult.npy", a)
