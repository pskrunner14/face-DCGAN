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
Project: https://github.com/pskrunner14/face-DCGAN
"""
import os
import click
import logging

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import (
    generator, 
    discriminator
)
from utils import (
    load_dataset, 
    sample_noise_batch, 
    iterate_minibatches
)

@click.command()
@click.option(
    '-nd',
    '--noise-dim',
    default=256,
    help='Dimension of noise (1-D Tensor) to feed the generator.'
)
@click.option(
    '-glr', 
    '--gen-lr', 
    default=0.001, 
    help='Learning rate for minimizing generator loss during training.'
)
@click.option(
    '-dlr', 
    '--disc-lr', 
    default=0.001, 
    help='Learning rate for minimizing discriminator loss during training.'
)
@click.option(
    '-bz',
    '--batch-size',
    default=64,
    help='Mini batch size to use during training.'
)
@click.option(
    '-ne', 
    '--num-epochs', 
    default=50, 
    help='Number of epochs for training models.'
)
@click.option(
    '-se',
    '--save-every',
    default=1,
    help='Epoch interval to save model checkpoints during training.'
)
@click.option(
    '-tb',
    '--tensorboard-vis',
    is_flag=True,
    help='Flag for TensorBoard Visualization.'
)
def train(noise_dim, gen_lr, disc_lr, batch_size, num_epochs, save_every, tensorboard_vis):
    """Trains the Deep Convolutional Generative Adversarial Network (DCGAN).

    Args:
        noise_dim (int, optional):
            Dimension of noise (1-D Tensor) to feed the generator.
            If unspecified, defaults to 256.
        gen_lr (float, optional): 
            Learning rate for minimizing generator loss during training. 
            If unspecified, defaults to 0.001.
        disc_lr (float, optional): 
            Learning rate for minimizing discriminator loss during training. 
            If unspecified, defaults to 0.001.
        batch_size (int, optional):
            Batch size of minibatches to use during training. 
            If unspecified, defauls to 32.
        num_epochs (int, optional):
            Number of epochs for training model. If unspecified, 
            defaults to 10.
        save_every (int, optional):
            Epoch interval to save model checkpoints during training. 
            If unspecified, defaults to 1.
        tensorboard_vis (bool, optional):
            Flag for TensorBoard Visualization. If unspecified, 
            defaults to `False`.
    """
    # Load Dataset.
    logging.info('loading LFW dataset into memory')
    X, IMAGE_SHAPE = load_dataset(dimx=36, dimy=36)

    tf.reset_default_graph()

    with tf.device('/gpu:0'):

        # Define placeholders for input data.
        noise = tf.placeholder('float32', [None, noise_dim])
        real_data = tf.placeholder('float32', [None, ] + list(IMAGE_SHAPE))

        # Create Generator and Discriminator models.
        logging.info('creating generator and discriminator')
        g_logits, g_out = generator(noise, train=True)
        d_probs, d_fake_logits = discriminator(g_out, train=True)
        d_probs2, d_real_logits = discriminator(real_data, train=True)

        # Define Generator(G) ops.
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))
        g_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr)
        g_vars = get_vars_by_scope('generator')
        g_train_step = g_optimizer.minimize(g_loss, var_list=g_vars)

        # Define Discriminator ops.
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_real_logits)))
        d_loss = d_loss_real + d_loss_fake
        d_optimizer = tf.train.AdamOptimizer(learning_rate=disc_lr)
        d_vars = get_vars_by_scope('discriminator')
        d_train_step = d_optimizer.minimize(d_loss, var_list=d_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start training.
    logging.info('training DCGAN model')
    for epoch in range(num_epochs):
        eval_noise = sample_noise_batch(16)
        idx = np.random.choice(range(X.shape[0]), size=16)
        eval_real_data = X[idx]
        for X_batch in tqdm(iterate_minibatches(X, batch_size, shuffle=True), 
                            total=X.shape[0] // batch_size, desc='Epoch[{}/{}]'
                            .format(epoch, num_epochs), leave=False):
            sess.run([d_train_step], feed_dict={real_data: X_batch,
                                                noise: sample_noise_batch(batch_size)})
            for _ in range(2):
                sess.run([g_train_step], feed_dict={noise: sample_noise_batch(batch_size)})
        
        # Generate images using G and save in `out/`.
        d_loss_iter, g_loss_iter, eval_images = sess.run([d_loss, g_loss, g_out], 
                                                        feed_dict={real_data: eval_real_data,
                                                                   noise: eval_noise})
        tl.visualize.save_images(eval_images, [4, 4], 'out/eval_{}.png'.format(epoch))
        logging.info('Epoch[{}/{}]    g_loss: {:.6f}   -   d_loss: {:.6f}'
                    .format(epoch, num_epochs, g_loss_iter, d_loss_iter))

    sess.close()

def get_vars_by_scope(scope_name):
    """ Returns list of trainable vars under scope name.

    Args:
        scope_name (str):
            Variable scope name.

    Returns:
        list of `tf.Variable`:
            List of trainable variables under given scope name.
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO')
    try:
        train()
    except KeyboardInterrupt:
        logging.info('EXIT')

if __name__ == '__main__':
    main()