import numpy as np
import os
import skimage
import skimage.transform

def random_rotated_image(images, low = -20, high = 20):
    image_num, image_channel, image_height, image_width = images.shape
    rotated_degree_1 = list(np.random.randint(low = low, high = 5, size = image_num // 2))
    rotated_degree_2 = list(np.random.randint(low = 5, high = high, size = image_num // 2))
    rotated_degree = np.array(rotated_degree_1 + rotated_degree_2)
    np.random.shuffle(rotated_degree)

    rotated_image = [skimage.transform.rotate(np.rollaxis(images[i], 0, 3), rotated_degree[i], mode="reflect") for i in range(image_num)]
    rotated_image = np.rollaxis(np.array(rotated_image), 3, 1)
    return np.array(rotated_image, dtype = np.float32)

data_dir = os.environ['CIFAR10_DIR']
test_images = np.array(np.load(os.path.join(data_dir, "cifar10TestingData.npy")).reshape(10000, 3, 32, 32), dtype=np.float32)
test_images = test_images / 255.0
test_labels = np.load(os.path.join(data_dir, "cifar10TestingDataLabel.npy"))

test_images = random_rotated_image(test_images)

np.save(os.path.join(data_dir, "X_test_rotated.npy"), test_images)
np.save(os.path.join(data_dir, "Y_test_rotated.npy"), test_labels)

