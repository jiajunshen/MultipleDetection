opyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns*depth]

    images = images.reshape(images.shape[0],
                          images.shape[1] * images.shape[2] * images.shape[3])
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   dtype=dtypes.float32):
 
    train_images = np.load(train_dir + "cifar10TrainingData.npy") / 255.0

    train_images = np.rollaxis(train_images, 1, 4)
    

    train_labels = np.load(train_dir + "cifar10TrainingDataLabel.npy")

    test_images = np.load(train_dir + "cifar10TestingData.npy") / 255.0

    test_images = np.rollaxis(test_images, 1, 4)
    
    test_labels = np.load(train_dr + "cifar10TestingDataLabel.npy")

    VALIDATION_SIZE = 5000    

    validation_images = train_images[:VALIDATION_SIZE]

    validation_labels = train_labels[:VALIDATION_SIZE]

    train_images = train_images[VALIDATION_SIZE:]

    train_labels = train_labels[VALIDATION_SIZE:]

    train = DataSet(train_images, train_labels, dtype=dtype)

    validation = DataSet(validation_images, validation_labels, dtype=dtype)

    test = DataSet(test_images, test_labels, dtype=dtype)

    return base.Datasets(train=train, validation=validation, test=test)


def load_mnist():
  return read_data_sets(os.environ['CIFAR10_DIR'])
