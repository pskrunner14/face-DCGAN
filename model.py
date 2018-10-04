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
import numpy as np
import tensorflow as tf

def create_generator(input_noise):
    
    with tf.variable_scope('generator') as global_scope:
        
        emb_size = input_noise.get_shape().as_list()[-1]

        with tf.variable_scope('dense_1') as scope:
            weights = tf.get_variable('weights', shape=[emb_size, 10 * 8 * 8], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[10 * 8 * 8], 
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            dense = tf.add(tf.matmul(input_noise, weights), bias)
            bn = tf.layers.batch_normalization(dense)
            dense_1 = tf.nn.leaky_relu(bn, alpha=0.2, name=scope.name)

        dense_1_reshaped = tf.reshape(dense_1, shape=[-1, 8, 8, 10], name='dense_1_reshaped')

        with tf.variable_scope('deconv_1') as scope:
            kernel = tf.get_variable('kernel', shape=[5, 5, 64, 10], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[64],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            _, input_H, input_W, _ = dense_1_reshaped.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(dense_1_reshaped, filter=kernel, 
                                            output_shape=[-1, input_H + 4, input_W + 4, 64],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            bn = tf.layers.batch_normalization(deconv)
            deconv_1 = tf.nn.leaky_relu(bn, alpha=0.2, name=scope.name)

        with tf.variable_scope('deconv_2') as scope:
            kernel = tf.get_variable('kernel', shape=[5, 5, 64, 64], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[64],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            _, input_H, input_W, _ = deconv_1.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(deconv_1, filter=kernel, 
                                            output_shape=[-1, input_H + 4, input_W + 4, 64],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            bn = tf.layers.batch_normalization(deconv)
            deconv_2 = tf.nn.leaky_relu(bn, alpha=0.2, name=scope.name)

        H, W = deconv_2.get_shape().as_list()[1: 3]
        upsampled_deconv_2 = tf.image.resize_nearest_neighbor(deconv_2, (2 * H, 2 * W), 
                                                                name='upsampled_deconv_2')

        with tf.variable_scope('deconv_3') as scope:
            kernel = tf.get_variable('kernel', shape=[7, 7, 3, 64], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[3],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            _, input_H, input_W, _ = upsampled_deconv_2.get_shape().as_list()
            deconv = tf.nn.conv2d_transpose(upsampled_deconv_2, filter=kernel, 
                                            output_shape=[-1, input_H + 4, input_W + 4, 3],
                                            strides=[1, 1, 1, 1], padding='VALID')
            deconv = tf.add(deconv, bias)
            deconv_3 = tf.layers.batch_normalization(deconv, name=global_scope.name)
            
        logits = deconv_3
        out = tf.nn.tanh(deconv_3, name='out')

        # with tf.variable_scope('deconv_4') as scope:
        #     kernel = tf.get_variable('kernel', shape=[3, 3, 10, 32], 
        #                             initializer=tf.random_uniform_initializer(), 
        #                             dtype=tf.float32)
        #     bias = tf.get_variable('bias', shape=[32],
        #                             initializer=tf.constant_initializer(0.0), 
        #                             dtype=tf.float32)
        #     B, input_H, input_W, _ = deconv_3.get_shape().as_list()
        #     deconv = tf.nn.conv2d_transpose(deconv_3, filter=kernel, 
        #                                     output_shape=[B, input_H + 4, input_W + 4, 32],
        #                                     strides=[1, 1, 1, 1], padding='VALID')
        #     deconv = tf.add(deconv, bias)
        #     bn = tf.layers.batch_normalization(deconv)
        #     deconv_4 = tf.nn.leaky_relu(bn, alpha=0.2)

        # with tf.variable_scope('deconv_5') as scope:
        #     kernel = tf.get_variable('kernel', shape=[3, 3, 10, 32], 
        #                             initializer=tf.random_uniform_initializer(), 
        #                             dtype=tf.float32)
        #     bias = tf.get_variable('bias', shape=[32],
        #                             initializer=tf.constant_initializer(0.0), 
        #                             dtype=tf.float32)
        #     B, input_H, input_W, _ = deconv_4.get_shape().as_list()
        #     deconv = tf.nn.conv2d_transpose(deconv_4, filter=kernel, 
        #                                     output_shape=[B, input_H + 4, input_W + 4, 32],
        #                                     strides=[1, 1, 1, 1], padding='VALID')
        #     deconv = tf.add(deconv, bias)
        #     bn = tf.layers.batch_normalization(deconv)
        #     deconv_5 = tf.nn.leaky_relu(bn, alpha=0.2, name=scope.name)

        # with tf.variable_scope('out') as scope:
        #     kernel = tf.get_variable('kernel', shape=[3, 3, 10, 3], 
        #                             initializer=tf.random_uniform_initializer(), 
        #                             dtype=tf.float32)
        #     bias = tf.get_variable('bias', shape=[3],
        #                             initializer=tf.constant_initializer(0.0), 
        #                             dtype=tf.float32)
        #     conv = tf.nn.conv2d(deconv_3, filter=kernel, 
        #                         strides=[1, 1, 1, 1],
        #                         padding='SAME')
        #     generator = tf.add(conv, bias, name=global_scope.name)

    return out, logits

def create_discriminator(image_data):

    with tf.variable_scope('discriminator') as global_scope:

        with tf.variable_scope('conv_1') as scope:
            kernel = tf.get_variable('kernel', shape=[3, 3, 3, 32], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[32],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            conv = tf.nn.conv2d(image_data, filter=kernel, 
                                strides=[1, 1, 1, 1], 
                                padding='SAME')
            conv = tf.add(conv, bias)
            l_relu = tf.nn.leaky_relu(conv)
            conv_1 = tf.layers.average_pooling2d(l_relu, pool_size=[2, 2], 
                                                strides=(1, 1), name=scope.name)

        with tf.variable_scope('conv_2') as scope:
            kernel = tf.get_variable('kernel', shape=[3, 3, 32, 32], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[32],
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            conv = tf.nn.conv2d(conv_1, filter=kernel, 
                                strides=[1, 2, 2, 1], 
                                padding='SAME')
            conv = tf.add(conv, bias)
            l_relu = tf.nn.leaky_relu(conv)
            conv_2 = tf.layers.average_pooling2d(l_relu, pool_size=[2, 2], 
                                                strides=(1, 1), name=scope.name)

        with tf.variable_scope('dense') as scope:
            dim = np.prod(conv_2.get_shape().as_list()[1: ])
            flattened = tf.reshape(conv_2, [-1, dim])
            weights = tf.get_variable('weights_1', shape=[dim, 256], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias_1', shape=[256], 
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            dense_1 = tf.add(tf.matmul(flattened, weights), bias)
            bn = tf.layers.batch_normalization(dense_1)
            l_relu = tf.nn.leaky_relu(bn, alpha=0.2, name=scope.name)
            weights = tf.get_variable('weights_2', shape=[256, 1], 
                                    initializer=tf.random_uniform_initializer(), 
                                    dtype=tf.float32)
            bias = tf.get_variable('bias_2', shape=[1], 
                                    initializer=tf.constant_initializer(0.0), 
                                    dtype=tf.float32)
            dense = tf.add(tf.matmul(l_relu, weights), bias, name=scope.name)
            
        logits = dense
        out = tf.nn.sigmoid(dense, name=global_scope.name)

    return out, logits
            