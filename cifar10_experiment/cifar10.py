import lasagne
import numpy as np
import theano.tensor as T
from PIL import Image
import theano
import os, sys, gzip
from six.moves import urllib
import tarfile
import pickle
import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10_input
import lasagne
from lasagne.layers import LocalResponseNormalization2DLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, InputLayer, DimshuffleLayer
from lasagne.regularization import regularize_layer_params_weighted, l2
#from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_integer('batch_size', 128,
#                            """Number of images to process in a batch.""")
#tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)

def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
    eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

    Raises:
    ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                                batch_size=FLAGS.batch_size)


def inference(input_var=None):
    """Build the CIFAR-10 model.

    Args:
    images: Images returned from distorted_inputs() or inputs().

    Returns:
    Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    input_layer = InputLayer((None, IMAGE_SIZE, IMAGE_SIZE, 3), input_var=input_var)
    
    shuffle_layer = DimshuffleLayer(input_layer, (0, 3, 1, 2))

    # conv1
    conv1 = Conv2DLayer(shuffle_layer, num_filters=64, filter_size=(5,5),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.HeNormal(),
                        b=lasagne.init.Constant(0.0),
                        name="conv1")
    
    # pool1
    pool1 = MaxPool2DLayer(conv1, pool_size=(3, 3), stride=(2, 2), pad=1)

    # norm1
    norm1 = LocalResponseNormalization2DLayer(pool1, alpha=0.001 / 9.0,
                                              beta=0.75, k=1.0, n=9)
    
    # conv2
    conv2 = Conv2DLayer(norm1, num_filters=64, filter_size=(5,5),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='same', W=lasagne.init.HeNormal(),
                        b=lasagne.init.Constant(0.1),
                        name='conv2')

    # norm2
    norm2 = LocalResponseNormalization2DLayer(conv2, alpha=0.001 / 9.0,
                                              beta=0.75, k=1.0, n=9)
    
    # pool2
    pool2 = MaxPool2DLayer(norm2, pool_size=(3, 3), stride=(2, 2), pad=1)
    
    # fc1
    fc1 = DenseLayer(pool2, num_units=384,
                     nonlinearity=lasagne.nonlinearities.rectify,
                     W=lasagne.init.HeNormal(), b=lasagne.init.Constant(0.1),
                     name="fc1")

    # fc2
    fc2 = DenseLayer(fc1, num_units=192,
                     nonlinearity=lasagne.nonlinearities.rectify,
                     W=lasagne.init.HeNormal(), b=lasagne.init.Constant(0.1),
                     name="fc2")

    # fc3
    softmax_layer = DenseLayer(fc2, num_units=10,
                               nonlinearity=lasagne.nonlinearities.softmax,
                               W=lasagne.init.HeNormal(),
                               b=lasagne.init.Constant(0.0),
                               name="softmax")

    # Weight Decay
    weight_decay_layers = {fc1: 0.004, fc2: 0.004}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return softmax_layer, l2_penalty


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    

