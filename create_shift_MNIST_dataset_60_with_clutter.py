import numpy as np
import cv2
import os

def load_data(trainingData, trainingLabel, testingData, testingLabel, dataset = "MNIST"):
    trainingData = os.environ[dataset] + trainingData
    trainingLabel = os.environ[dataset] + trainingLabel
    testingData = os.environ[dataset] + testingData
    testingLabel = os.environ[dataset] + testingLabel

    X_train = np.array(np.load(trainingData), dtype = np.float32).reshape(-1, 1, 28, 28)
    Y_train = np.array(np.load(trainingLabel), dtype = np.uint8)
    X_test = np.array(np.load(testingData), dtype = np.float32).reshape(-1, 1, 28, 28)
    Y_test = np.array(np.load(testingLabel), dtype = np.uint8)

    return X_train, Y_train, X_test, Y_test
def rotateImage(image, angle):
    if len(image.shape) == 3:
            image = image[0]
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    return np.array(result[np.newaxis, :, :], dtype = np.float32)


def extend_image(inputs, images = None, mega_patch_w=8, size=40, num_strokes=5):
    if len(inputs.shape) == 3:
        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])

    if images is None:
        images = inputs[np.random.randint(low = 0, high = inputs.shape[0], size = 100)]

    num_samples = inputs.shape[0]
    image_size, image_channel, image_w, image_h = images.shape

    extended_images_with_clutter = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)

    margin_size = (size - inputs.shape[2]) / 2
    for i in range(inputs.shape[0]):

        for j in range(num_strokes):
            intensity = 0
            t_y = np.random.randint(size - mega_patch_w)
            t_x = np.random.randint(size - mega_patch_w)
            patch = np.zeros((mega_patch_w, mega_patch_w))
            while(intensity < 10):
                index = np.random.randint(image_size)
                s_y = np.random.randint(image_w - mega_patch_w)
                s_x = np.random.randint(image_h - mega_patch_w)
                patch = images[index, 0, s_x:s_x + mega_patch_w, s_y:s_y + mega_patch_w]
                intensity = np.sum(patch)
            extended_images_with_clutter[i, 0, t_x:t_x + mega_patch_w, t_y:t_y + mega_patch_w] = patch
        
        #margin_x = np.random.randint(0, size - inputs.shape[2])
        #margin_y = np.random.randint(0, size - inputs.shape[3])
        x_value_index = np.where(np.sum(inputs[i, 0], axis = 1) != 0)[0]
        y_value_index = np.where(np.sum(inputs[i, 0], axis = 0) != 0)[0]

        x_left = x_value_index[0]
        x_right = x_value_index[-1]
        x_length = x_right - x_left + 1
        y_left = y_value_index[0]
        y_right = y_value_index[-1]
        y_length = y_right - y_left + 1

        margin_x = np.random.randint(0, size - x_length)
        margin_y = np.random.randint(0, size - y_length)

        extended_images[i, :, margin_x:margin_x + x_length, margin_y:margin_y + y_length] = inputs[i, :, x_left: x_left + x_length, y_left: y_left + y_length]
        extended_images_with_clutter[i, :, margin_x:margin_x + x_length, margin_y:margin_y + y_length] = inputs[i, :, x_left : x_left + x_length, y_left: y_left + y_length]
    return extended_images, extended_images_with_clutter




X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
#X_test = extend_image(X_test, 40)
#X_train = extend_image(X_train, 40)

train_size = y_train.shape[0]
all_images = []
all_labels = []
for j in range(1):
    angles_1 = list(np.random.randint(low = -90, high = 0, size = train_size // 2))
    angles_2 = list(np.random.randint(low = 0, high = 90, size = train_size // 2))
    angles = np.array(angles_1 + angles_2)
    np.random.shuffle(angles)
    rotated_image = np.array([rotateImage(X_train[i], angles[i]) for i in range(train_size)], dtype = np.float32)
    all_images.append(rotated_image)
    all_labels.append(y_train)
#all_images = np.vstack(all_images)
#all_labels = np.hstack(all_labels)
all_images = X_train
all_labels = y_train

print(all_images.shape, all_labels.shape)

index = np.arange(1 * train_size)
np.random.shuffle(index)

all_images = all_images[index]
all_labels = all_labels[index]



x_train, x_train_cluttered = extend_image(all_images, size = 60)
y_train = all_labels

test_size = y_test.shape[0]
all_images = []
all_labels = []
for j in range(1):
    angles_1 = list(np.random.randint(low = -90, high = 0, size = test_size // 2))
    angles_2 = list(np.random.randint(low = 0, high = 90, size = test_size // 2))
    angles = np.array(angles_1 + angles_2)
    np.random.shuffle(angles)
    rotated_image = np.array([rotateImage(X_test[i], angles[i]) for i in range(test_size)], dtype = np.float32)
    all_images.append(rotated_image)
    all_labels.append(y_test)
#all_images = np.vstack(all_images)
all_images = X_test
#all_labels = np.hstack(all_labels)
all_labels = y_test

print(all_images.shape, all_labels.shape)

index = np.arange(1 * test_size)
np.random.shuffle(index)

all_images = all_images[index]
all_labels = all_labels[index]



x_test, x_test_cluttered  = extend_image(all_images, size = 60)
y_test = all_labels

np.savez("./data/mnist_60_shift_cluttered.npz", x_train = x_train, y_train = y_train, x_test = x_test, y_test=y_test, x_train_cluttered = x_train_cluttered, x_test_cluttered = x_test_cluttered)

