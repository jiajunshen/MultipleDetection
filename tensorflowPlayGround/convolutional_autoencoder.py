from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from tensorflow.examples.tutorials.mnist import input_data

from flags import FLAGS

def loss_x_entropy(output, target):
  """Cross entropy loss
  See https://en.wikipedia.org/wiki/Cross_entropy
  Args:
    output: tensor of net output
    target: tensor of net we are trying to reconstruct
  Returns:
    Scalar tensor of cross entropy
  """
  with tf.name_scope("xentropy_loss"):
      net_output_tf = tf.convert_to_tensor(output, name='input')
      target_tf = tf.convert_to_tensor(target, name='target')
      cross_entropy = tf.add(tf.mul(tf.log(net_output_tf, name='log_output'),
                                    target_tf),
                             tf.mul(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
      return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                 name='xentropy_mean')

# Use relu as activation for the convolution layers
# The encoder procedure use padding = 'valid', the decoder procedure use padding = 'same'
def conv2d(x, W, b, padding = 'SAME', stride = 1):
    print(x.get_shape(), W.get_shape(), b.get_shape())
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding = padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.sigmoid(x)

def dense_layer(x, W, b):
    return tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b))


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

def convAutoencoder(x, weights, bias, weights_key, bias_key):
    
    x = tf.reshape(x, shape = [-1, 28, 28, 1])
    print(weights_key, bias_key) 
    #encoder procedure
    encoder_conv_1 = conv2d(x, weights[weights_key[0]], bias[bias_key[0]])
    encoder_conv_2 = conv2d(encoder_conv_1, weights[weights_key[1]], bias[bias_key[1]])
    encoder_pool_1 = maxpool2d(encoder_conv_2)
    encoder_conv_3 = conv2d(encoder_pool_1, weights[weights_key[2]], bias[bias_key[2]])
    encoder_pool_2 = maxpool2d(encoder_conv_3)
    print(encoder_pool_2.get_shape())
    encoder_pool_2_reshape = tf.reshape(encoder_pool_2, shape = [-1, 1568])
    encoder_dense_1 = dense_layer(encoder_pool_2_reshape, weights[weights_key[3]], bias[bias_key[3]])


    #decoder_procedure
    decoder_dense_1 = dense_layer(encoder_dense_1, weights[weights_key[4]], bias[bias_key[4]])
    decoder_dense_1_reshape = tf.reshape(decoder_dense_1, shape = [-1, 7, 7, 32])
    decoder_upscale_2 = upscale2d(decoder_dense_1_reshape, [1, 2], scale = 2)
    decoder_conv_3 = conv2d(decoder_upscale_2, weights[weights_key[5]], bias[bias_key[5]])
    decoder_upscale_1 = upscale2d(decoder_conv_3, [1, 2], scale = 2)
    decoder_conv_2 = conv2d(decoder_upscale_1, weights[weights_key[6]], bias[bias_key[6]])
    decoder_conv_1 = conv2d(decoder_conv_2, weights[weights_key[7]], bias[bias_key[7]])
    
    print(decoder_conv_1.get_shape())
    #output
    output = tf.reshape(decoder_conv_1, shape = [-1, 784])

    print(output.get_shape())
    
    return output

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    learning_rate = 0.000003
    training_epochs = 20
    batch_size = 100
    n_input = 784
    n_train = mnist.train.num_examples
    weights = {
        'encode_conv_weight_1':tf.Variable(tf.random_normal([3, 3, 1, 16])),
        'encode_conv_weight_2':tf.Variable(tf.random_normal([3, 3, 16, 16])),
        'encode_conv_weight_3':tf.Variable(tf.random_normal([3, 3, 16, 32])),
        'encode_dense_weight_1':tf.Variable(tf.random_normal([1568, 16])),
        'decode_dense_weight_1':tf.Variable(tf.random_normal([16, 1568])),
        'decode_conv_weight_3': tf.Variable(tf.random_normal([3, 3, 32, 16])),
        'decode_conv_weight_2': tf.Variable(tf.random_normal([3, 3, 16, 16])),
        'decode_conv_weight_1': tf.Variable(tf.random_normal([3, 3, 16, 1]))
    }

    bias = {
        'encode_conv_bias_1':tf.Variable(tf.random_normal([16])),
        'encode_conv_bias_2':tf.Variable(tf.random_normal([16])),
        'encode_conv_bias_3':tf.Variable(tf.random_normal([32])),
        'encode_dense_bias_1':tf.Variable(tf.random_normal([16])),
        'decode_dense_bias_1':tf.Variable(tf.random_normal([1568])),
        'decode_conv_bias_3': tf.Variable(tf.random_normal([16])),
        'decode_conv_bias_2': tf.Variable(tf.random_normal([16])),
        'decode_conv_bias_1': tf.Variable(tf.random_normal([1]))
    }
    
    weights_key = ['encode_conv_weight_1',
                    'encode_conv_weight_2',
                    'encode_conv_weight_3',
                    'encode_dense_weight_1',
                    'decode_dense_weight_1',
                    'decode_conv_weight_3',
                    'decode_conv_weight_2', 
                    'decode_conv_weight_1']
    
    bias_key = ['encode_conv_bias_1',
                    'encode_conv_bias_2',
                    'encode_conv_bias_3',
                    'encode_dense_bias_1',
                    'decode_dense_bias_1',
                    'decode_conv_bias_3',
                    'decode_conv_bias_2', 
                    'decode_conv_bias_1']

    x = tf.placeholder(tf.float32, [batch_size, n_input])

    reconstruction = convAutoencoder(x, weights, bias, weights_key, bias_key)
    loss = loss_x_entropy(reconstruction, x) 
    #train_step = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999).minimize(loss)

    init = tf.initialize_all_variables()
   

    sess = tf.Session()
 
    summary_dir = pjoin(FLAGS.summary_dir, 'conv_auto_encoder_training')
    summary_writer = tf.train.SummaryWriter(summary_dir,
                                            graph_def=sess.graph_def,
                                            flush_secs=FLAGS.flush_secs)
    
    print("\n\n")
    print("| Training Step | Cross Entropy |   Epoch  |")
    print("|---------------|---------------|----------|")

    sess.run(init)
    step = 1
    while step * batch_size < training_epochs * n_train:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {x:batch_x}
        sess.run(train_step, feed_dict = feed_dict) 
        if step % 100 == 0:
            loss_summary = sess.run(loss, feed_dict = feed_dict)
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
