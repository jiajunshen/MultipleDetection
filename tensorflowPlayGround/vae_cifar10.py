from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

from os.path import join as pjoin

from flags import FLAGS

import cifar10
import math

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



def createVAE(x, weights, bias, weightsKey, biasKey):
    z_mean, z_log_sigma_sq = encoder_network(x, weights, bias, weightsKey, biasKey)
    eps = tf.random_normal((100, 500), 0, 1, dtype = tf.float32)
    
    z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) 

    reconstruction = decoder_network(z, weights, bias, weightsKey, biasKey)
   
    
    recon_loss = -tf.reduce_sum(x * tf.log(1e-10 + reconstruction) + (1 - x) * tf.log(1e-10 + 1 - reconstruction), 1)
    
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    
    cost = tf.reduce_mean(recon_loss + latent_loss)
    #if (math.isnan(cost)):
    #    print(recon_loss, tf.reduce_mean(recon_loss))
    #    print("====================================")
    #    print(latent_loss, tf.reduce_mean(latent_loss))
    #    print("====================================")
    #    print(reconstruction)
    return reconstruction, cost, tf.reduce_mean(recon_loss), recon_loss, latent_loss


def encoder_network(x, weights, bias, weightsKey, biasKey):
    print(x.get_shape()) 
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[weightsKey[0]]), bias[biasKey[0]]))
    print(layer_1.get_shape())
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights[weightsKey[1]]), bias[biasKey[1]]))
    #print(layer_2.get_shape())
    z_mean = tf.add(tf.matmul(layer_1, weights[weightsKey[1]]), bias[biasKey[1]])
    print(z_mean.get_shape())
    z_log_sigma_sq = tf.add(tf.matmul(layer_1, weights[weightsKey[2]]), bias[biasKey[2]])
    print(z_log_sigma_sq.get_shape())

    return (z_mean, z_log_sigma_sq)

def decoder_network(x, weights, bias, weightsKey, biasKey):
    
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[weightsKey[3]]), bias[biasKey[3]]))
    reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights[weightsKey[4]]), bias[biasKey[4]]))
    
    #reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights[weightsKey[6]]), bias[biasKey[6]]))

    return reconstruction





def main():
    cifar10_data = cifar10.load_cifar10()
    learning_rate = 0.001
    training_epochs = 1000
    batch_size = 100
    n_input = 3072
    n_train = cifar10_data.train.num_examples


    weightsKey = ['encode_h1', 'encode_h2','encode_out_mean', 'encode_out_log_sigma', 'decode_h1', 'decode_h2', 'decode_out_mean', 'decode_out_log_sigma']
    biasKey = ['encode_h1', 'encode_h2','encode_out_mean', 'encode_out_log_sigma', 'decode_h1', 'decode_h2', 'decode_out_mean', 'decode_out_log_sigma']
    
    weights = {

        #Encoder procedure
        weightsKey[0]: tf.Variable(xavier_init(3072, 1024)),
        #weightsKey[1]: tf.Variable(xavier_init(1024, 1024)),
        # Latent space is 16
        weightsKey[1]: tf.Variable(xavier_init(1024, 500)),
        weightsKey[2]: tf.Variable(xavier_init(1024, 500)),

        #Decode procedure
        weightsKey[3]: tf.Variable(xavier_init(500, 1024)),
        #weightsKey[5]: tf.Variable(xavier_init(1024, 1024)),
        weightsKey[4]: tf.Variable(xavier_init(1024, 3072)),
        #weightsKey[7]: tf.Variable(xavier_init(500, 784))
    }
    
    bias = {

        #Encoder procedure
        biasKey[0]: tf.Variable(tf.zeros([1024], dtype = tf.float32)),
        #biasKey[1]: tf.Variable(tf.zeros([1024], dtype = tf.float32)),
        # Latent space is 16
        biasKey[1]: tf.Variable(tf.zeros([500], dtype = tf.float32)),
        biasKey[2]: tf.Variable(tf.zeros([500], dtype = tf.float32)),

        #Decode procedure
        #biasKey[4]: tf.Variable(tf.zeros([1024], dtype = tf.float32)),
        biasKey[3]: tf.Variable(tf.zeros([1024], dtype = tf.float32)),
        biasKey[4]: tf.Variable(tf.zeros([3072], dtype = tf.float32)),
        #biasKey[7]: tf.Variable(xavier_init(500, 784))
    }

    x = tf.placeholder(tf.float32, [batch_size, n_input])
    reconstruction, cost, recon_loss, recon_full_loss, latent_loss = createVAE(x, weights, bias, weightsKey, biasKey)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    summary_dir = pjoin(FLAGS.summary_dir, 'vae_cifar10')
    summary_writer = tf.train.SummaryWriter(summary_dir, graph_def = sess.graph_def, flush_secs = FLAGS.flush_secs)
    sess.run(init)

    step = 1
    
    print("\n\n")
    print("| Training Step | Cross Entropy |   Epoch  |")
    print("|---------------|---------------|----------|")
    
    while step * batch_size < training_epochs * n_train:
        batch_x, batch_y = cifar10_data.train.next_batch(batch_size)
        feed_dict = {x:batch_x}
        sess.run(optimizer, feed_dict = feed_dict)
        reconstruction_, cost_, recon_, recon_full_, latent_full_ = sess.run([reconstruction, cost, recon_loss, recon_full_loss, latent_loss], feed_dict = feed_dict)
        if (math.isnan(float(cost_))):
            print(reconstruction_, np.mean(reconstruction_))
            print("==================================")
            print(cost_)
            print("==================================")
            print(recon_)
            print("==================================")
            print(recon_full_, np.mean(recon_full_))
            print("==================================")
            print(latent_full_, np.mean(latent_full_))
            print("==================================")
    
        if step % 450 == 0:
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
                                         32,
                                         32, 3)),
                             max_images=FLAGS.batch_size)
            reconstruction_summary_op = \
                tf.image_summary("reconstruction_image",
                            tf.reshape(reconstruction, 
                                        (FLAGS.batch_size,
                                         32,
                                         32, 3)),
                             max_images=FLAGS.batch_size)

            summary_img_str = sess.run(image_summary_op,
                                   feed_dict=feed_dict)
            summary_writer.add_summary(summary_img_str, step)

            summary_recon_image_str = sess.run(reconstruction_summary_op,
                                    feed_dict = feed_dict)
            summary_writer.add_summary(summary_recon_image_str, step)
        step+=1
    
    batch_x, batch_y = cifar10_data.test.next_batch(FLAGS.batch_size)
    feed_dict = {x: np.array(batch_x)}

    recon_target = sess.run(reconstruction, feed_dict=feed_dict)
    np.save("vae_cifar10.npy", recon_target.reshape(100, 32, 32, 3))

    print("Optimization Finished!")


if __name__ == "__main__":
    main()
