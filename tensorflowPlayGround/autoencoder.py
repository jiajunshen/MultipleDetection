from __future__ import division, print_function
import time
import os.path import join as pjoin
import tensorflow as tf
import numpy as np
import math
import random
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
      net_output_tf = tf.ops.convert_to_tensor(output, name='input')
      target_tf = tf.ops.convert_to_tensor(target, name='target')
      cross_entropy = tf.add(tf.mul(tf.log(net_output_tf, name='log_output'),
                                    target_tf),
                             tf.mul(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
      return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                 name='xentropy_mean')


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
                  name_b_out = self._bias_str.format(i + 1) + "_out"

                  b_shape = (self.__shape[i],)
                  b_init = tf.zeros(b_shape)
                  self[name_b_out] = tf.Variable(b_init,
                                                 trainable=True,
                                                 name=name_b_out)


    def _w(self, n, suffix = ""):
        return self[self._weights_str.format(n)+suffix]

    def _b(self, n, suffix = ""):
        return self[self._bias_str.format(n) + suffix]

    def get_variable_to_init(self,n):
        assert n > 0
        assert n <= self._num_hidden_layers + 1

        vars_to_init = [self._w(n), self._b(n)]

        if n <= self._num_hidden_layers:
            vars_to_init.append(self._b(n, "_out"))

        if 1 < n <= self._num_hidden_layers:
            vars_to_init.append(self._w(n -1, "_fixed"))
            vars_to_init.append(self._b(n - 1, "_fixed"))


    def _activate(x, w, b, transpose_w = False):
        y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b = transpose_w), b))
        return y

    def net(self, input_pl, n, is_target = False):
        """
        Args:
            input_pl: placeholder of AE inputs
            n:        input specifying pretrain step
            is_target: bool specifying if required tensor should be the tar
        """        
        last_output = input_pl
        for i in xrange(n - 1):
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")
            last_output = self._activate(last_output, w, b)
        

        out = self._activate(last_output, self._w(n), self._b(n, "_out"))

        out = tf.maximum(out, 1.e-9)
        out = tf.minimum(out, 1 - 1.e-9)

        return out
    


    def main_unsupervised():
      with tf.Graph().as_default() as g:
        sess = tf.Session()

        num_hidden = FLAGS.num_hidden_layers

        # The shape of the hidden layers
        ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                            for j in xrange(num_hidden)]

        # The shape of the all the layers, including the input layer and the classification layer... Why do we need the classification layer again?
        ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes + [FLAGS.num_classes]

        ae = AutoEncoder(ae_shape, sess)


        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        num_train = data.train.num_examples

        learning_rates = {j: getattr(FLAGS,
                                     "pre_layer{0}_learning_rate".format(j + 1))
                          for j in xrange(num_hidden)}

        noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1))
                 for j in xrange(num_hidden)}

        with tf.variable_scope("net_training".format(n)):

            input_ = tf.placeholder(dtype=tf.float32,
                                    shape=(FLAGS.batch_size, ae_shape[0]),
                                    name='ae_input_pl')
            
            target_for_loss = ae.net(input_, n)


            #loss = loss_x_entropy(layer, target_for_loss)
            loss = tf.sqrt(tf.reduce_mean(tf.square(input_ - target_for_loss)))


        


            summary_dir = pjoin(FLAGS.summary_dir, 'pretraining_{0}'.format(n))
            summary_writer = tf.train.SummaryWriter(summary_dir,
                                                    graph_def=sess.graph_def,
                                                    flush_secs=FLAGS.flush_secs)
            summary_vars = [ae["biases{0}".format(n)], ae["weights{0}".format(n)]]

            hist_summarries = [tf.histogram_summary(v.op.name, v)
                               for v in summary_vars]
            
            summary_op = tf.merge_summary(hist_summarries)

            vars_to_init = ae.get_variables_to_init(n)
            vars_to_init.append(global_step)
            sess.run(tf.initialize_variables(vars_to_init))

            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

            print("\n\n")
            print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
            print("|---------------|---------------|---------|----------|")

            for step in xrange(FLAGS.pretraining_epochs * num_train):
              feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])

              loss_summary, loss_value = sess.run([train_op, loss],
                                                  feed_dict=feed_dict)

              if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                image_summary_op = \
                    tf.image_summary("training_images",
                                     tf.reshape(input_,
                                                (FLAGS.batch_size,
                                                 FLAGS.image_size,
                                                 FLAGS.image_size, 1)),
                                     max_images=FLAGS.batch_size)

                summary_img_str = sess.run(image_summary_op,
                                           feed_dict=feed_dict)
                summary_writer.add_summary(summary_img_str)

                output = "| {0:>13} | {1:13.4f} | Layer {2} | Epoch {3}  |"\
                         .format(step, loss_value, n, step // num_train + 1)

                print(output)

      return ae



if __name__ == '__main__':
    ae = main_unsupervised()
