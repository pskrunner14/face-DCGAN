""" Deep Convolutional Generative Adversarial Network (DCGAN).

Using deep convolutional generative adversarial networks (DCGAN) 
to generate face images from a noise distribution.

References:
    - Generative Adversarial Nets. Goodfellow et al. arXiv: 1406.2661.
    - Unsupervised Representation Learning with Deep Convolutional 
    Generative Adversarial Networks. A Radford, L Metz, S Chintala. 
    arXiv: 1511.06434.

Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434.pdf).

Author: Prabhsimran Singh
Project: https://github.com/pskrunner14/face-DCGAN
"""

import numpy as np
import tensorflow as tf

from ops import (
    dense_layer,
    conv_layer,
    deconv_layer
)

def generator(input_noise, train=True):
    """ Creates convolutional generator model.
        
    See https://arxiv.org/abs/1511.06434.pdf.

    Args:
        input_noise (tf.placeholder): Input noise distribution tensor. 
        train (bool, optional): Flag for whether to freeze batch-norm layer vars. If unspecified, defaults to `True`.
    Returns:
        Tensor containing images generated from the noise distribution.
    """
    dense_1_shape = [8, 8, 10]
    dense_1_units = np.prod(dense_1_shape)
    
    # We need to pass `batch_size` for using in `output_shape` in deconv op.
    # See https://riptutorial.com/tensorflow/example/29767/using-tf-nn-conv2d-transpose-for-arbitary-batch-sizes-and-with-automatic-output-shape-calculation-
    batch_size = tf.shape(input_noise)[0]

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        dense_1 = dense_layer(input_noise, train, units=dense_1_units, name='dense_1')
        dense_1_reshaped = tf.reshape(dense_1, shape=[-1, ] + dense_1_shape, name='dense_1_reshaped')
        deconv_1 = deconv_layer(dense_1_reshaped, train, kernel_dims=(5, 5), 
                                in_channels=dense_1_shape[-1], out_channels=64, 
                                batch_size=batch_size, name='deconv_1')
        deconv_2 = deconv_layer(deconv_1, train, kernel_dims=(5, 5), 
                                in_channels=64, out_channels=64, 
                                batch_size=batch_size, name='deconv_2')
        # H, W = deconv_2.get_shape().as_list()[1: 3]
        # upsampled_deconv_2 = tf.image.resize_nearest_neighbor(deconv_2, (2 * H, 2 * W), name='upsampled_deconv_2')
        upsampled_deconv_2 = tf.keras.layers.UpSampling2D(size=(2, 2))(deconv_2)
        deconv_3 = deconv_layer(upsampled_deconv_2, train, kernel_dims=(7, 7), 
                                in_channels=64, out_channels=32,
                                batch_size=batch_size, name='deconv_3')
        logits = conv_layer(deconv_3, train, kernel_dims=(3, 3), in_channels=32, 
                            out_channels=3, name='logits', padding='VALID', 
                            use_avgpool=False, use_batchnorm=False, activation=None)
        out = tf.nn.tanh(logits, name=scope.name)
    return out

def discriminator(image_data, train=True):
    """ Creates convolutional discriminator model.
        
    See https://arxiv.org/abs/1511.06434.pdf.

    Args:
        image_data (tf.placeholder): Tensor containing real/fake images to classify.
        train (bool, optional): Flag for whether to freeze batch-norm layer vars. If unspecified, defaults to `True`.
    Returns:
        Tensors containing probabilites and logits pertaining to input images being real/fake.
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        conv_1 = conv_layer(image_data, train, kernel_dims=(3, 3), 
                            in_channels=3, out_channels=32, name='conv_1')
        conv_2 = conv_layer(conv_1, train, kernel_dims=(3, 3), 
                            in_channels=32, out_channels=32, name='conv_2', 
                            strides=(2, 2))
        dim = np.prod(conv_2.get_shape().as_list()[1: ])
        flattened_1 = tf.reshape(conv_2, [-1, dim])
        dense_1 = dense_layer(flattened_1, train, 256, name='dense_1')
        logits = dense_layer(dense_1, train, 1, name='logits', 
                            use_batchnorm=False, activation=None)
        probs = tf.nn.sigmoid(logits, name=scope.name)
    return probs, logits
