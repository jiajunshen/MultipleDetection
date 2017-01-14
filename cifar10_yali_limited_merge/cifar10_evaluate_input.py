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
import skimage
import skimage.transform
from six.moves import xrange  # pylint: disable=redefined-builtin

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

import collections

class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 test=False,
                 distortion=True,
                 rangeType = "int",
                 dtype=np.float32):
        self._num_examples = images.shape[0]
        self._images = np.array(images, dtype=dtype)
        if rangeType == "int":
            self._images = self._images / 255.0
        self._labels = np.array(labels, dtype=np.int32)
        self._index_in_epoch = 0
        self._index_in_eval_epoch = 0
        self._distortion = distortion
        
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_eval_batch(self, batch_size, image_size=IMAGE_SIZE, distort=False):
        start = self._index_in_eval_epoch
        self._index_in_eval_epoch += batch_size
        if start >= NUM_EXAMPLES_PER_EPOCH_FOR_EVAL:
            self._index_in_eval_epoch = 0
            return None, None
        else:
            end = self._index_in_eval_epoch
            return self._images[start:end], self._labels[start:end]
            

def read_evaluation_data_sets(data_dir, dtype=np.float32):
    test_images = np.array(np.load(os.path.join(data_dir, "cifar10TestingData.npy")).reshape(10000, 3, 32, 32), dtype=dtype)
    test_labels = np.load(os.path.join(data_dir, "cifar10TestingDataLabel.npy"))

    test = DataSet(test_images, test_labels, test=True)

    Datasets = collections.namedtuple('Datasets', ['test'])

    return Datasets(test = test)

def read_evaluation_rotated_data_sets(data_dir, dtype=np.float32):
    test_images = np.array(np.load(os.path.join(data_dir, "X_test_rotated.npy")).reshape(10000, 3, 32, 32), dtype=dtype)
    test_labels = np.load(os.path.join(data_dir, "Y_test_rotated.npy"))

    test = DataSet(test_images, test_labels, test=True)

    Datasets = collections.namedtuple('Datasets', ['test'], rangeType = "real")

    return Datasets(test = test)

def load_cifar10_test():
    return read_evaluation_data_sets(os.environ['CIFAR10_DIR'])

def load_cifar10_rotated_test():
    return read_evaluation_rotated_data_sets(os.environ['CIFAR10_DIR'])


