from CNNForMnist import build_cnn, load_data
import numpy as np
import cv2
import amitgroup as ag

def extend_image(inputs, size = 32):
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    margin_size = (32 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images



X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

# Create test_file
if 0:
    X_test = extend_image(X_test, 32)

    X_test = np.array(X_test.reshape(X_test.shape[0], 32, 32), dtype = np.float64)

    test_size = y_test.shape[0]
    all_images = []
    all_labels = []

    imdef = ag.util.DisplacementFieldWavelet((32, 32), 'db8')

    for j in range(5):
        deformed_image = np.array([imdef.randomize(0.01).deform(X_test[i]) for i in range(test_size)], dtype = np.float32)
        all_images.append(deformed_image)
        all_labels.append(y_test)
    all_images = np.vstack(all_images)
    all_labels = np.hstack(all_labels)

    print(all_images.shape, all_labels.shape)

    index = np.arange(5 * test_size)
    np.random.shuffle(index)

    all_images = all_images[index, 2: 30, 2:30]
    all_labels = all_labels[index]


    np.save("/home/jiajun/.mnist/X_test_deformed.npy", all_images)
    np.save("/home/jiajun/.mnist/Y_test_deformed.npy", all_labels)

if 1:

    X_train = extend_image(X_train, 32)

    X_train = np.array(X_train.reshape(X_train.shape[0], 32, 32), dtype = np.float64)

    train_size = y_train.shape[0]

    imdef = ag.util.DisplacementFieldWavelet((32, 32), 'db8')

    deformed_image = np.array([imdef.randomize(0.01).deform(X_train[i]) for i in range(train_size)], dtype = np.float32)
    all_images = deformed_image
    all_labels = y_train

    print(all_images.shape, all_labels.shape)

    #index = np.arange(5 * test_size)
    #np.random.shuffle(index)

    all_images = all_images[:, 2: 30, 2:30]
    #all_labels = all_labels[index]


    np.save("/home/jiajun/.mnist/X_train_deformed.npy", all_images)
    np.save("/home/jiajun/.mnist/Y_train_deformed.npy", all_labels)

