from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from tensorflow.examples.tutorials.mnist import input_data

from flags import FLAGS


mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -low
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

class VariationalAutoencoder(object):
    def __init__(self, network_architecture, transfer_fct = tf.nn.softplus, learning_rate = 0.001, batch_size = 100):
        self.weights = {}
        self._create_network()
        self._create_loss_optimizer()
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


def conv2d(x, W, b, padding = 'SAME', stride = 1, activation = tf.nn.softplus):
    print(x.get_shape(), W.get_shape(), b.get_shape())
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding = padding)
    x = tf.nn.bias_add(x, b)
    return activation(x)

def dense_layer(x, W, b):
    return tf.nn.softplus(tf.nn.bias_add(tf.matmul(x, W), b))


def maxpool2d(x, stride = 2):
    #Ksize is the window size, stride is the sliding window size
    return tf.nn.max_pool(x, ksize = [1, stride, stride, 1], strides = [1, stride, stride,1], padding = 'SAME')

def repeat(x, dimension, scale = 2):
    x_shape = x.get_shape()
    result_shape = tf.TensorShape(x_shape)
    result_shape.dims[dimension] = result_shape.dims[dimension] * scale
    intermediate_shape = tf.TensorShape(x_shape)
    intermediate_shape.dims.insert(dimension + 1, tf.Dimension(scale))

    broadCastArray = tf.ones(intermediate_shape)

    new_x_shape = tf.TensorShape(x_shape)

    new_x_shape.dims.insert(dimension + 1, tf.Dimension(1))
    x = tf.reshape(x, new_x_shape)

    result = x * broadCastArray
    result = tf.reshape(result, result_shape)
    return result 


def upscale2d(x, dimension, scale = 2):
    # Here we would assume the x is 4d with shape [n, w, h, c]. After scaling, we would have the result of shape [n, w * scale, h * scale, c]
    for i in range(len(dimension)):
        x = repeat(x, dimension[i], scale)
    return x

def createVAE(x, weights, bias, weightsKey, biasKey):
    x_reshape = tf.reshape(x, shape = [-1, 28, 28, 1])
    z_mean, z_log_sigma_sq = encoder_network(x_reshape, weights, bias, weightsKey, biasKey)
    eps = tf.random_normal((100, 50), 0, 1, dtype = tf.float32)
    
    z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) 

    reconstruction = decoder_network(z, weights, bias, weightsKey, biasKey)
    
    recon_loss = -tf.reduce_sum(x * tf.log(1e-10 + reconstruction) + (1 - x) * tf.log(1e-10 + 1 - reconstruction), 1)
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    cost = tf.reduce_mean(recon_loss + latent_loss)

    return reconstruction, cost, tf.reduce_mean(recon_loss)


def encoder_network(x, weights, bias, weights_key, bias_key):
    encoder_conv_1 = conv2d(x, weights[weights_key[0]], bias[bias_key[0]])
    encoder_conv_2 = conv2d(encoder_conv_1, weights[weights_key[1]], bias[bias_key[1]])
    encoder_pool_1 = maxpool2d(encoder_conv_2)
    encoder_conv_3 = conv2d(encoder_pool_1, weights[weights_key[2]], bias[bias_key[2]])
    encoder_pool_2 = maxpool2d(encoder_conv_3)
    encoder_pool_2_reshape = tf.reshape(encoder_pool_2, shape = [-1, 1568])
    z_mean = tf.add(tf.matmul(encoder_pool_2_reshape, weights[weights_key[3]]), bias[bias_key[3]])
    z_log_sigma_sq = dense_layer(encoder_pool_2_reshape, weights[weights_key[4]], bias[bias_key[4]])

    return (z_mean, z_log_sigma_sq)

def decoder_network(x, weights, bias, weights_key, bias_key):
    
    #decoder_procedure
    decoder_dense_1 = dense_layer(x, weights[weights_key[5]], bias[bias_key[5]])
    decoder_dense_1_reshape = tf.reshape(decoder_dense_1, shape = [-1, 7, 7, 32])
    decoder_upscale_2 = upscale2d(decoder_dense_1_reshape, [1, 2], scale = 2)
    decoder_conv_3 = conv2d(decoder_upscale_2, weights[weights_key[6]], bias[bias_key[6]])
    decoder_upscale_1 = upscale2d(decoder_conv_3, [1, 2], scale = 2)
    decoder_conv_2 = conv2d(decoder_upscale_1, weights[weights_key[7]], bias[bias_key[7]])
    decoder_conv_1 = conv2d(decoder_conv_2, weights[weights_key[8]], bias[bias_key[8]], activation = tf.nn.sigmoid)
    
    reconstruction = tf.reshape(decoder_conv_1, shape = [-1, 784])

    return reconstruction





def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    learning_rate = 0.0003
    training_epochs = 20
    batch_size = 100
    n_input = 784
    n_train = mnist.train.num_examples

    boundary = 6.0 / (32 + 32)
    
    weights = {
        'encode_conv_weight_1':tf.Variable(tf.random_uniform((3, 3, 1, 16), minval = -boundary, maxval = boundary)),
        'encode_conv_weight_2':tf.Variable(tf.random_uniform((3, 3, 16, 16), minval = -boundary, maxval = boundary)),
        'encode_conv_weight_3':tf.Variable(tf.random_uniform((3, 3, 16, 32), minval = -boundary, maxval = boundary)),
        'encode_output_mean':tf.Variable(xavier_init(1568, 50)),
        'encode_output_log_sigma': tf.Variable(xavier_init(1568, 50)),
        'decode_dense_weight_1':tf.Variable(xavier_init(50, 1568)),
        'decode_conv_weight_3': tf.Variable(tf.random_uniform((3, 3, 32, 16), minval = -boundary, maxval = boundary)),
        'decode_conv_weight_2': tf.Variable(tf.random_uniform((3, 3, 16, 16), minval = -boundary, maxval = boundary)),
        'decode_conv_weight_1': tf.Variable(tf.random_uniform((3, 3, 16, 1), minval = -boundary, maxval = boundary))
    }

    bias = {
        'encode_conv_bias_1':tf.Variable(tf.zeros([16])),
        'encode_conv_bias_2':tf.Variable(tf.zeros([16])),
        'encode_conv_bias_3':tf.Variable(tf.zeros([32])),
        'encode_output_mean':tf.Variable(tf.zeros([50])),
        'encode_output_log_sigma': tf.Variable(tf.zeros([50])),
        'decode_dense_bias_1':tf.Variable(tf.zeros([1568])),
        'decode_conv_bias_3': tf.Variable(tf.zeros([16])),
        'decode_conv_bias_2': tf.Variable(tf.zeros([16])),
        'decode_conv_bias_1': tf.Variable(tf.zeros([1]))
    }
    
    weights_key = ['encode_conv_weight_1',
                    'encode_conv_weight_2',
                    'encode_conv_weight_3',
                    'encode_output_mean',
                    'encode_output_log_sigma',
                    'decode_dense_weight_1',
                    'decode_conv_weight_3',
                    'decode_conv_weight_2', 
                    'decode_conv_weight_1']
    
    bias_key = ['encode_conv_bias_1',
                    'encode_conv_bias_2',
                    'encode_conv_bias_3',
                    'encode_output_mean',
                    'encode_output_log_sigma',
                    'decode_dense_bias_1',
                    'decode_conv_bias_3',
                    'decode_conv_bias_2', 
                    'decode_conv_bias_1']

    x = tf.placeholder(tf.float32, [batch_size, n_input])
    reconstruction, cost, recon_loss = createVAE(x, weights, bias, weights_key, bias_key)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    summary_dir = pjoin(FLAGS.summary_dir, 'variational autoencoder')
    summary_writer = tf.train.SummaryWriter(summary_dir, graph_def = sess.graph_def, flush_secs = FLAGS.flush_secs)
    sess.run(init)

    step = 1
    
    print("\n\n")
    print("| Training Step | Cross Entropy |   Epoch  |")
    print("|---------------|---------------|----------|")
    
    while step * batch_size < training_epochs * n_train:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {x:batch_x}
        sess.run(optimizer, feed_dict = feed_dict)
        
        if step % 100 == 0:
            loss_summary = sess.run(recon_loss, feed_dict = feed_dict)
            #loss_summary_op = tf.scalar_summary("reconstruction_error", loss_summary)
            
            #summary_scalar_str = sess.run(loss_summary_op, feed_dict = {x: batch_x})
            #summary_writer.add_summary(summary_scalar_str, step)
                
            output = "| {0:>13} | {1:13.4f} | Epoch {2}  |"\
                 .format(step, loss_summary, step * batch_size // n_train + 1)

            print(output)
            

        if step % 1100 == 0:
            image_summary_op = \
                tf.image_summary("training_images",
                             tf.reshape(x,
                                        (FLAGS.batch_size,
                                         FLAGS.image_size,
                                         FLAGS.image_size, 1)),
                             max_images=FLAGS.batch_size)
            reconstruction_summary_op = \
                tf.image_summary("reconstruction_image",
                            tf.reshape(reconstruction, 
                                        (FLAGS.batch_size,
                                         FLAGS.image_size,
                                         FLAGS.image_size, 1)),
                             max_images=FLAGS.batch_size)

            summary_img_str = sess.run(image_summary_op,
                                   feed_dict=feed_dict)
            summary_writer.add_summary(summary_img_str, step)

            summary_recon_image_str = sess.run(reconstruction_summary_op,
                                    feed_dict = feed_dict)
            summary_writer.add_summary(summary_recon_image_str, step)
        step+=1

    print("Optimization Finished!")


if __name__ == "__main__":
    main()
