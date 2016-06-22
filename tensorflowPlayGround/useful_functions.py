from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -low
    return tf.random_uniform((fan_int, fan_out), minval = low, maxval = high, dtype = tf.float32)



def weight_variable(shape):
    initial_value = xavier_init(shape[0], shape[1])
    return tf.Variable(initial_value)

def conv_weight_variable(shape):

    boundary = 6.0 / (2 * np.max(shape[0], shape[3]))
    initial_value = tf.random_uniform(shape, minval = -boundary, maxval = boundary)

    return tf.Variable(initial_value)


def bias_variable(shape):
    initial_value = tf.constant(0, shape = shape)
    return tf.Variable(initial_value)


def variable_summaries(var, name):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary("mean/" + name, mean)
        std = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary("std/" + name, std)
        tf.scalar_summary("max/" + name, tf.reduce_max(var))
        tf.scalar_summary("min/" + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def fully_connected(input_tensor, input_dim, output_dim, layer_name, activation = tf.nn.relu, dropout = False, dropoutProb = 0.5):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + "/weights")
        biases = bias_variable([output_dim])
        variable_summaries(biases, layer_name + "/biases")
        pre_activation = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + "/preactivation", pre_activation)
        after_activation = activation(pre_activation, "activation")
        tf.histogram_summary(layer_name + "/after_activation", after_activation)

        if dropout:
            keep_prob = tf.constant(dropoutProb)
            after_activation = tf.nn.dropout(after_activation, keep_prob)
    return after_activation



def conv2d(input_tensor, filter_shape, num_filter, layer_name, padding = 'SAME', activation = tf.nn.relu, dropout = False, dropoutProb = 0.5):
    with tf.name_scope(layer_name):
        weights = conv_weight_variable(filter_shape + (num_filter,))
        biases = bias_variable([num_filter]) 
        pre_activation = tf.nn.conv2d(input_tensor, weights, strides = [1, 1, 1, 1], padding = padding)
        pre_activation = tf.nn.bias_add(pre_activation, biases)
        after_activation = activation(pre_activation, "activation")

        if dropout:
            keep_prob = tf.constant(dropoutProb)
            after_activation = tf.nn.dropout(after_activation, keep_prob)
    return after_activation
        














            
