imagenet_directory = "/hdd/Documents/Data/ImageNet2011/"
name_index_list = ["n02691156_plane", "n02958343_car", "n04530566_boat",\
                   "n02924116_bus", "n04090263_gun", "n04256520_sofa",\
                   "n04379243_table", "n03001627_chair", "n03211117_display"]

from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt
import colorsys
from PIL import Image, ImageEnhance

train_image_list = []
train_label_list = []
test_image_list = []
test_label_list = []
for item in name_index_list:
    current_directory = join(imagenet_directory, item)
    img_path = [join(current_directory, f) for f in listdir(current_directory) if isfile(join(current_directory, f))]
    current_index = name_index_list.index(item)
    for file_path in img_path[:1000]:
        object_image = misc.imresize(misc.imread(file_path), (32, 32, 3))
        if len(object_image.shape) == 2:
            object_image = np.repeat(object_image.reshape(32, 32, 1), 3, axis = 2)
        train_image_list.append(object_image)
        train_label_list.append(current_index)
    for file_path in img_path[1000:1200]:
        object_image = misc.imresize(misc.imread(file_path), (32, 32, 3))
        if len(object_image.shape) == 2:
            object_image = np.repeat(object_image.reshape(32, 32, 1), 3, axis = 2)
        test_image_list.append(object_image)
        test_label_list.append(current_index)

train_image_list = np.asarray(train_image_list)
train_image_label = np.asarray(train_label_list)
test_image_list = np.asarray(test_image_list)
test_image_label = np.asarray(test_label_list)

np.save("/hdd/Documents/Data/ShapeNetCoreV1/imagenet_train.npy", train_image_list)
np.save("/hdd/Documents/Data/ShapeNetCoreV1/imagenet_train_label.npy", train_image_label)
np.save("/hdd/Documents/Data/ShapeNetCoreV1/imagenet_test.npy", test_image_list)
np.save("/hdd/Documents/Data/ShapeNetCoreV1/imagenet_test_label.npy", test_image_label)
