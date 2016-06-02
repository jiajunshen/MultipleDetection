import tensorflow as tf
import numpy as np
import math
import random


class AutoEncoder(object):
    _weights_str = "weights{0}"
    _bias_str = "bias{0}"
    def __init__(self, shape, sess):
        """
        Autoencoder Initizalizer:
            shape: a list of layer shape we need to specify
            sess: the tensorflow session we are going to use
        """
        self._shape = shape
        self._num_hidden_layers = len(self._shape) - 2
        
        self._variables = {}
        self._session = sess
        self._setup_variables()



    @property
    def shape(self):
        return self._shape
    
    @property
    def num_hidden_layers(self):
        return self._num_hidden_layers
    
    @property
    def session(self):
        return self._session

    def __getitem__(self, item):
        return self._variables[item]

    def __setitem__(self, key, value):
        self._variables[key] = value

    #https://github.com/cmgreen210/TensorFlowDeepAutoencoder/blob/master/code/ae/autoencoder.py
    #http://cmgreen.io/2016/01/04/tensorflow_deep_autoencoder.html
    def _setup_variables(self):
        with tf.name_scope("autoencoder_variables"):
            for i in range(self._num_hidden_layers + 1):
                name_w = self._weights_str.format(i + 1)
                w_shape = (self._shape[i], self._shape[i + 1])
                # We use xavier initializer here
                initializer_bound = tf.mul(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
                w_init = tf.random_uniform(w_shape, -1 * initializer_bound, 1 * initializer_bound)
                self[name_w] = tf.Variable(w_init, name = name_w, trainable = True)

                name_b = self._bias_str.format(i + 1)
                b_shape = (self._shape[i], )
                b_init = tf.zeros(b_shape)
                self[name_b] = tf.Variable(b_init, name = name_b, trainable = True)
        
                
                ## Why do we need the following? we will see

                if i < self.__num_hidden_layers:
                  # Hidden layer fixed weights (after pretraining before fine tuning)
                  self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                                                        name=name_w + "_fixed",
                                                        trainable=False)

                  # Hidden layer fixed biases
                  self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                                                        name=name_b + "_fixed",
                                                        trainable=False)

                  # Pretraining output training biases
                  name_b_out = self._biases_str.format(i + 1) + "_out"
                  b_shape = (self.__shape[i],)
                  b_init = tf.zeros(b_shape)
                  self[name_b_out] = tf.Variable(b_init,
                                                 trainable=True,
                                                 name=name_b_out)
















def loss_cross_entropy(output, target):
    output_tf = tf.ops.convert_to_tensor(output, name = "net_output")
    target_tf = tf.ops.convert_to_tensor(target, name = "target")
    cross_entropy = tf.add(tf.mul(tf.log(output_tf), target_tf), tf.mul(tf.log( 1- output_tf), 1 - target_tf))
    return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name = 'cross_entropy_mean')


def conv_relu(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape, initializer = tf.random_normal_initializer())
    bias = tf.get_variable("bias", bias_shape, initializer = tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides = [1, 1, 1, 1], padding = 'SAME')
    return tf.nn.relu(conv + bias)


def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        relu2 = conv_relu(relu1, [5, 5, 32, 32], [32])
    
def run():
    with tf.variable_scope("image_filters") as scope:
        result1 = my_image_filter(image1)
        scope.reuse_variables()
    





def simple_test():
    sess = tf.Session()
    x = tf.placeholder("float", [None, 4])
    autoencoder = create(x, [2])
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(autoencoder['cost'])


    # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
    c1 = np.array([0,0,0.5,0])
    c2 = np.array([0.5,0,0,0])

    # do 1000 training steps
    for i in range(2000):
        # make a batch of 100:
        batch = []
        for j in range(100):
            # pick a random centroid
            if (random.random() > 0.5):
                vec = c1
            else:
                vec = c2
            batch.append(np.random.normal(vec, 0.1))
        sess.run(train_step, feed_dict={x: np.array(batch)})
        if i % 100 == 0:
            print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})


if __name__ == '__main__':
    simple_test()
