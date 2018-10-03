""" Deep Convolutional Generative Adversarial Network (DCGAN).
Using deep convolutional generative adversarial networks (DCGAN) to generate
face images from a noise distribution.
References:
    -Generative Adversarial Nets.
    Goodfellow et al. arXiv: 1406.2661.
    - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. 
    A Radford, L Metz, S Chintala. arXiv: 1511.06434.
Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf)
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).
Author: Prabhsimran Singh
Project: https://github.com/pskrunner14/faceGAN
"""

import tensorflow as tf

class FaceGAN():

    def __init__(self):
        pass

class Generator():

    def __init__(self):
        self.__EMB_SIZE = 256
        self.out = self.__create_model()

    def __create_model(self):
        
        noise = tf.placeholder(dtype=tf.float32, shape=[None, self.__EMB_SIZE], name='noise')
        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob_gen')

        with tf.variable_scope('dense_1') as scope:
            weights = tf.get_variable('weights', shape=[self.__EMB_SIZE, 10 * 8 * 8], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[10 * 8 * 8], 
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            dense = tf.add(tf.matmul(noise, weights), bias)
            bn = tf.layers.batch_normalization(dense)
            dense_1 = tf.nn.leaky_relu(bn, alpha=0.2, name=scope.name)

        dense_1_reshaped = tf.reshape(dense_1, shape=[-1, 8, 8, 10], name='dense_1_reshaped')

        with tf.variable_scope('deconv_1') as scope:
            kernel = tf.get_variable('kernel', shape=[5, 5, 10, 64], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[64],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            B, input_H, input_W, _ = dense_1_reshaped.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(dense_1_reshaped, filter=kernel, 
                                            output_shape=[B, input_H + 4, input_W + 4, 64],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            bn = tf.layers.batch_normalization(deconv)
            l_relu = tf.nn.leaky_relu(bn, alpha=0.2)
            deconv_1 = tf.nn.dropout(l_relu, keep_prob=keep_prob, name=scope.name)

        with tf.variable_scope('deconv_2') as scope:
            kernel = tf.get_variable('kernel', shape=[5, 5, 10, 64], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[64],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            B, input_H, input_W, _ = deconv_1.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(deconv_1, filter=kernel, 
                                            output_shape=[B, input_H + 4, input_W + 4, 64],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            bn = tf.layers.batch_normalization(deconv)
            l_relu = tf.nn.leaky_relu(bn, alpha=0.2)
            deconv_2 = tf.nn.dropout(l_relu, keep_prob=keep_prob, name=scope.name)

        H, W = deconv_2.get_shape().as_list()[1: 3]
        upsampled_deconv_2 = tf.image.resize_nearest_neighbor(deconv_2, (2 * H, 2 * W), 
                                                                name='upsampled_deconv_2')

        with tf.variable_scope('deconv_3') as scope:
            kernel = tf.get_variable('kernel', shape=[3, 3, 10, 32], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[32],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            B, input_H, input_W, _ = upsampled_deconv_2.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(upsampled_deconv_2, filter=kernel, 
                                            output_shape=[B, input_H + 4, input_W + 4, 32],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            bn = tf.layers.batch_normalization(deconv)
            l_relu = tf.nn.leaky_relu(bn, alpha=0.2)
            deconv_3 = tf.nn.dropout(l_relu, keep_prob=keep_prob, name=scope.name)

        with tf.variable_scope('deconv_4') as scope:
            kernel = tf.get_variable('kernel', shape=[3, 3, 10, 32], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[32],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            B, input_H, input_W, _ = deconv_3.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(deconv_3, filter=kernel, 
                                            output_shape=[B, input_H + 4, input_W + 4, 32],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            bn = tf.layers.batch_normalization(deconv)
            l_relu = tf.nn.leaky_relu(bn, alpha=0.2)
            deconv_4 = tf.nn.dropout(l_relu, keep_prob=keep_prob, name=scope.name)

        with tf.variable_scope('deconv_5') as scope:
            kernel = tf.get_variable('kernel', shape=[3, 3, 10, 32], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[32],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            B, input_H, input_W, _ = deconv_4.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(deconv_4, filter=kernel, 
                                            output_shape=[B, input_H + 4, input_W + 4, 32],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            bn = tf.layers.batch_normalization(deconv)
            deconv_5 = tf.nn.leaky_relu(bn, alpha=0.2, name=scope.name)

        with tf.variable_scope('out') as scope:
            kernel = tf.get_variable('kernel', shape=[3, 3, 10, 3], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[3],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            conv = tf.nn.conv2d(deconv_5, filter=kernel, 
                                strides=[1, 1, 1, 1], 
                                padding='SAME')
            out = tf.add(conv, bias, name='out')

        return out

class Discriminator():

    def __init__(self):
        self.__IMAGE_SIZE = (36, 36, 3)
        self.out = self.__create_model()
    
    def __create_model(self):
        pass