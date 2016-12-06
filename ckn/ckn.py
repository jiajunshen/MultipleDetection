import numpy as np
import os

import tensorflow as tf

class CKN_Layer():
    def __init__(self, num_features, part_shape, alpha, settings={}):
        tf.reset_default_graph()
        self._num_features = num_features
        self._part_shape = part_shape
        self._alpha = alpha
        self._z = None
        self._loss = None
        self._settings = settings
        self._sess = tf.Session()
        
    def extract(self, X, batch_size = 500):
        assert self._z is not None, "Must be trained before calling extract"
        X_size, X_w, X_h, X_channel = X.shape
        total_dimension = np.prod(self._part_shape)
        
        coded_result = np.zeros((X_size, X_w - self._part_shape[0] + 1, X_h - self._part_shape[1] + 1, self._num_features))
        
        X_input = tf.placeholder("float32", [None, total_dimension + 1], name = "X_input")
        fc_1 = tf.matmul(X_input, tf.transpose(self._z, [1, 0], name = "transpose"))
        
        code_function = tf.exp(fc_1)
        
        for i in range(X_size // batch_size):
            end = min((i + 1) * batch_size, X_size)
            start = i * batch_size
            X_select = X[start: end]
            code_x = np.ones((end-start, coded_result.shape[1], coded_result.shape[2], total_dimension + 1))
            norm_x = np.zeros((end-start, coded_result.shape[1], coded_result.shape[2]))
            for m in range(coded_result.shape[1]):
                for n in range(coded_result.shape[2]):
                    selected_patches = X_select[:,m:m+self._part_shape[0], n:n+self._part_shape[1], :].reshape(-1, total_dimension)
                    patches_norm = np.sqrt(np.sum(selected_patches ** 2, axis = 1))
                    patches_norm = np.clip(patches_norm, a_min = 0.00001, a_max = 10)
                    code_x[:, m, n, :total_dimension] = np.array([selected_patches[k] / patches_norm[k] for k in range(end-start)])
                    norm_x[:, m, n] = patches_norm
            code_x = code_x.reshape(-1, total_dimension + 1)
            feed_dict = {}
            feed_dict[X_input] = code_x
            tmp_coded_result = self._sess.run(code_function, feed_dict = feed_dict)
            reshape_result = tmp_coded_result.reshape(end-start, coded_result.shape[1], coded_result.shape[2], self._num_features)
            coded_result[start:end] = np.rollaxis(np.rollaxis(reshape_result, 3, 0) / norm_x, 0, 4)
            print norm_x

        return coded_result
        
        
    def get_patches(self, image, samples_per_image, patch_size):
        threshold = self._settings.get("threshold", 0.15)
        fr = 2
        the_patches = []

        w, h = [image.shape[i] - patch_size[i] + 1 for i in range(2)]

        for sample in range(samples_per_image):
            for tries in range(20):
                x, y = random.randint(0, w-1), random.randint(0, h-1)
                selection = [slice(x, x + patch_size[0]), slice(y, y + patch_size[1])]
                # Return grayscale patch and edges patch
                edgepatch = image[selection]
                #edgepatch_nospread = edges_nospread[selection]
                num = edgepatch[fr:-fr,fr:-fr].sum()
                if num >= threshold: 
                    the_patches.append(edgepatch)
                    break

        return np.array(the_patches)

    
    def setup_training(self, X):
        
        sample_per_image = self._settings.get('sample_per_image', 5)
        num_pairs = self._settings.get('num_pairs', 100000)
        total_dimension = np.prod(self._part_shape)
        
        sampled_patches = np.vstack([self.get_patches(X[i], sample_per_image, self._part_shape) for i in range(X.shape[0])])
        normalized_patches = np.array([sampled_patches[i] / np.sqrt(np.sum(sampled_patches[i]**2)) for i in range(sampled_patches.shape[0])])

        x = normalized_patches.reshape((-1, total_dimension))[:num_pairs]
        x_prime = normalized_patches.reshape((-1, total_dimension))[num_pairs:2 * num_pairs]
        
        print(x.shape)

        x_tilde = np.ones((num_pairs, total_dimension + 1))
        x_tilde[:, :total_dimension] = x
        x_prime_tilde = np.ones((num_pairs, total_dimension + 1))
        x_prime_tilde[:, :total_dimension] = x_prime

        G = np.zeros((total_dimension + 1, total_dimension + 1))
        for i in range(num_pairs):
            G += np.matmul(transpose(x_tilde[i:i+1]), x_tilde[i:i+1])
        G = G / num_pairs

        eigenvalue, eigenvector = np.linalg.eig(G)

        R = np.matmul(np.matmul(eigenvector, \
                                (np.identity(total_dimension + 1) * ((eigenvalue + np.mean(eigenvalue))**(-0.5)))\
                               ),\
                      transpose(eigenvector))
        self._z = tf.Variable(tf.random_normal([self._num_features, total_dimension + 1], 
                                                       mean = 0.0, stddev = self._alpha**2 / 4), dtype = "float32")
        R = np.array(R, dtype = np.float32)
        x_tilde = np.array(x_tilde, dtype=np.float32)
        x_prime_tilde = np.array(x_prime_tilde, dtype=np.float32)

        loss_1 = tf.exp(-np.sum((x_tilde - x_prime_tilde) ** 2, axis = 1) / (2 * self._alpha**2))

        matmul_result = tf.add(tf.matmul(tf.matmul(self._z, R), np.swapaxes(x_tilde, 0, 1)),
                              tf.matmul(tf.matmul(self._z, R), np.swapaxes(x_prime_tilde, 0, 1)))

        loss_2 = tf.reduce_sum(tf.exp(matmul_result),
                               reduction_indices=0)

        loss = tf.square(loss_1 - loss_2)
        self._loss = tf.reduce_mean(loss, reduction_indices=0)
        self._loss_2 = loss_2
        self._total_loss = loss

        
    def trained(self):
        return self._z is not None

    def train(self, X, Y=None):
        self.setup_training(X)
        num_iterations = self._settings.get("num_iterations", 0)
        learning_rate = self._settings.get("learning_rate", 0.0001)
        self.train_from_pairs(num_iterations, learning_rate)


    def train_from_pairs(self, num_iterations = 0, learning_rate = 0.0001):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self._loss)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        save_location = self._settings.get("save_location", "")
        
        if os.path.isfile(save_location):
            saver.restore(self._sess, save_location)
        else:
            self._sess.run(init)
        
        for i in range(num_iterations):
            _, loss_value, loss_2_value, loss_final_value = self._sess.run([train_step, self._loss, self._loss_2, self._total_loss])
            if i % 100 == 0:
                print(loss_value, loss_2_value, loss_final_value)
                print("================")
            saver.save(self._sess, save_location)



def main():
    X = np.load("/Users/jiajunshen/.mnist/X_train.npy")

    # First Layer
    ckn_layer_1 = CKN_Layer(50, (5, 5), 0.7, settings={"learning_rate": 0.001,
                                                       "num_iterations": 10000,
                                                       "num_pairs":30000,
                                                       "threshold":0.15,
                                                       "save_location": "./layer_1_new.ckpt"})
    ckn_layer_1 .train(X)
    output_first_layer = ckn_layer_1.extract(X.reshape(-1, 28, 28, 1))

    # Second Layer
    ckn_layer_2 = CKN_Layer(200, (2, 2), 0.5, settings={"learning_rate": 0.001, 
                                                        "num_iterations": 10000, 
                                                        "num_pairs":30000,
                                                        "threshold":0,
                                                        'sample_per_image': 50,
                                                        "save_location": "./layer_2_new.ckpt"})
    ckn_layer_2.train(output_first_layer)
    output_second_layer = ckn_layer_2.extract(output_first_layer)
    
