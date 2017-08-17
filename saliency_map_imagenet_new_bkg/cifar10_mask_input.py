# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 3
global NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 180000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 18000
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 66
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 34

import cifar10_mask as cifar10
import theano
import theano.tensor as T
import collections
import lasagne
from scipy import misc
from scipy import interpolate
train_dir = "./cifar10_theano_train_adam"

def random_crop(images, dims):
    image_num, image_channel, image_height, image_width = images.shape
    assert image_height > dims[1] and image_width > dims[2] and image_channel == dims[0]
    limit_height = image_height - dims[1] + 1
    limit_width = image_width - dims[2] + 1
    offset_height = np.random.randint(low = 0, high = limit_height, size = image_num) % limit_height
    offset_width = np.random.randint(low = 0, high = limit_width, size = image_num) % limit_width
    cropped_image = [images[i, :, offset_height[i]:offset_height[i]+dims[1],
                            offset_width[i]:offset_width[i]+dims[2]]
                     for i in range(image_num)]
    return np.array(cropped_image)


def random_flip_left_right(images):
    image_num, image_channel, image_height, image_width = images.shape
    uniform_random = np.random.uniform(size = image_num)
    mirror = uniform_random > 0.5
    mirrored_image = [images[i, :, :, ::(mirror[i] *2 - 1)]
                     for i in range(image_num)]
    return np.array(mirrored_image)

def random_brightness(images, max_delta):
    image_num, image_channel, image_height, image_width = images.shape
    delta = np.random.uniform(-max_delta, max_delta, size = image_num)
    adjusted_image = np.array([images[i] + delta[i] for i in range(image_num)])
    adjusted_image = np.clip(adjusted_image, a_min = 0, a_max = 255)
    return adjusted_image

def random_contrast(images, lower, upper):
    image_num, image_channel, image_height, image_width = images.shape
    contrast_factor = np.random.uniform(lower, upper, size = image_num)
    adjusted_image = [[(images[i, j, :, :] - np.mean(images[i, j, :, :])) * \
                       contrast_factor[i] + np.mean(images[i, j, :, :]) for j in range(3)] \
                      for i in range(image_num)]
    adjusted_image = np.clip(adjusted_image, a_min = 0, a_max = 255)
    return np.array(adjusted_image)

def per_image_standardization(images):
    image_num, image_channel, image_height, image_width = images.shape
    standardized_image = np.array([(images[i] - np.mean(images[i])) / \
                                   max(np.std(images[i]), 1.0 / np.sqrt(image_channel * \
                                                                        image_height * \
                                                                        image_width)) \
                                   for i in range(image_num)])
    return standardized_image


def resize_with_crop_or_pad(images, image_size):
    image_num, image_channel, image_height, image_width = images.shape
    offset = np.abs(image_size - image_height) // 2
    if image_size <= image_height:
        final_image = np.array([images[i, :, offset:offset + image_size, offset: offset + image_size] for i in range(image_num)])
    else:
        images = np.rollaxis(images, 1, 4)
        print(images.shape)
        final_image = [cv2.copyMakeBorder(images[i], offset, offset, offset, offset, cv2.BORDER_REFLECT) for i in range(image_num)]
        final_image = np.rollaxis(np.array(final_image), 3, 1)
    return final_image

def feature_extraction(data):
    image_input_var = T.tensor4('original_inputs')
    layer_result  = cifar10.build_cnn(image_input_var)
    model_output = [lasagne.layers.get_output(layer_result[i], deterministic=True) for i in range(len(layer_result))]

    if os.path.isfile(os.path.join(train_dir, 'latest_model.txt')):
        weight_file = ""
        with open(os.path.join(train_dir, 'latest_model.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        final_weights = [model_weights[i] for i in range(len(model_weights) - 2)]
        lasagne.layers.set_all_param_values(layer_result[-1], final_weights)

    extract_fn = theano.function([image_input_var], model_output)
    result_list = []
    for i in range(data.shape[0] // 5000 + 1):
        current_data = data[i * 5000: min(i * 5000 + 5000, data.shape[0])]
        extracted_result = extract_fn(current_data)
        current_example_feature_list = []
        #current_example_feature_list.append(current_data)
        for j in range(len(layer_result)):
            if j == 1 or j == 3:
                continue
            extracted_feature = extracted_result[j]
            print(extracted_feature.shape)
            if len(extracted_feature.shape) == 2:
                batch_feature = np.ones((extracted_feature.shape[0], IMAGE_SIZE, IMAGE_SIZE, extracted_feature.shape[1]))
                extracted_feature = extracted_feature[:,np.newaxis, np.newaxis, :]
                batch_feature = np.repeat(np.repeat(extracted_feature, IMAGE_SIZE, axis = 1), IMAGE_SIZE, axis = 2)
            else:
                batch_feature = []
                for m in range(extracted_feature.shape[0]):
                    feature_result_list = []
                    for n in range(extracted_feature.shape[1]):
                        interpx = np.linspace(0, extracted_feature[m,n].shape[0]-1, extracted_feature[m,n].shape[0])
                        interpy = np.linspace(0, extracted_feature[m,n].shape[1]-1, extracted_feature[m,n].shape[1])
                        f = interpolate.interp2d(interpx, interpy, extracted_feature[m,n],kind="linear")
                        interpNewX = np.linspace(0, extracted_feature[m,n].shape[0]-1, IMAGE_SIZE)
                        interpNewY = np.linspace(0, extracted_feature[m,n].shape[1]-1, IMAGE_SIZE)
                        feature_result = f(interpNewX, interpNewY)
                        feature_result_list.append(feature_result.reshape(1, IMAGE_SIZE, IMAGE_SIZE))
                    feature_result_list = np.vstack(feature_result_list)
                    feature_result_list = np.rollaxis(feature_result_list, 0, 3).reshape(1, IMAGE_SIZE, IMAGE_SIZE, -1)
                    batch_feature.append(feature_result_list)
                batch_feature = np.vstack(batch_feature)
            current_example_feature_list.append(np.rollaxis(batch_feature, 3, 1))
        current_example_feature = np.concatenate(current_example_feature_list, axis = 1)
        np.save(os.path.join(os.environ['SHAPENET'], "ShapeNet_Synthetic_texture_extracted_feature_%i.npy" %i), current_example_feature)
        result_list.append(current_example_feature)
    result_list = np.vstack(result_list)

    return result_list


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 test=False,
                 distortion=True,
                 dtype=np.float32):
        self._num_examples = images.shape[0]
        self._images = np.array(images, dtype=dtype)
        self._labels = np.array(labels, dtype=np.int32)
        self._index_in_epoch = 0
        self._index_in_eval_epoch = 0
        self._distortion = distortion

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def data_prepration(self, images, image_size, distortion=True):
        if distortion:
            cropped_image = random_crop(images, [3, image_size, image_size])
            mirrored_image = random_flip_left_right(cropped_image)
            brightness_adjusted_image = random_brightness(mirrored_image, 64)
            contrast_adjuested_image = random_contrast(brightness_adjusted_image, 0.2, 1.8)
            before_standardized_image = contrast_adjuested_image
        else:
            before_standardized_image = resize_with_crop_or_pad(images, image_size)
        standardized_image = np.array(per_image_standardization(before_standardized_image), dtype=np.float32)
        return standardized_image

    def next_batch(self, batch_size, image_size=IMAGE_SIZE):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        #return self.data_prepration(self._images[start:end], image_size, self._distortion),\
        #       self._labels[start:end]
        return self._images[start:end], self._labels[start:end], start

    def next_eval_batch(self, batch_size, image_size=IMAGE_SIZE, distort=False):
        start = self._index_in_eval_epoch
        self._index_in_eval_epoch += batch_size
        if start >= NUM_EXAMPLES_PER_EPOCH_FOR_EVAL:
            self._index_in_eval_epoch = 0
            return None, None
        else:
            end = self._index_in_eval_epoch
            #resize_image = resize_with_crop_or_pad(self._images[start:end], image_size)
            #standardized_image = per_image_standardization(resize_image)
            #return np.array(standardized_image, dtype=np.float32), self._labels[start:end]
            return self._images[start:end], self._labels[start:end]




def read_data_sets(data_dir, distortion=True, dtype=np.float32, training_num = 180000):
    global NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = training_num
    
    train_image = np.array(np.load(os.path.join(data_dir, "real_background_10_class_train.npy")).reshape(-1, 32, 32, 3), dtype=dtype)
    train_image = np.rollaxis(train_image, 3, 1)
    train_labels = np.array(np.load(os.path.join(data_dir, "mask_10_class_train.npy")).reshape(-1, 32 * 32),dtype=dtype) > 0

    total_num_train = train_image.shape[0]
    random_index = np.arange(total_num_train)
    np.random.shuffle(random_index)
    train_image = train_image[random_index]
    train_labels = train_labels[random_index]
    
    train_image = train_image[:NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
    train_labels = train_labels[:NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
    print(train_image.shape, train_labels.shape)

    test_image = np.array(np.load(os.path.join(data_dir, "real_background_10_class_test.npy")).reshape(-1, 32, 32, 3), dtype=dtype)
    test_image = np.rollaxis(test_image, 3, 1)
    test_labels = np.array(np.load(os.path.join(data_dir, "mask_10_class_test.npy")).reshape(-1, 32 * 32), dtype=dtype) > 0

    print(test_image.shape, test_labels.shape)

    train = DataSet(train_image, train_labels, distortion=distortion)
    test = DataSet(test_image, test_labels, test=True)

    Datasets = collections.namedtuple('Datasets', ['train', 'test'])

    return Datasets(train = train, test = test)

def load_cifar10(training_num):
    return read_data_sets(os.environ['SHAPENETV1'], training_num)
