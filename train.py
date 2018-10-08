import os
import click
import coloredlogs
import logging

import numpy as np
import tensorflow as tf
import tensorlayer as tl

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

    See https://arxiv.org/abs/1511.06434 for more details.

    Args: optional arguments [python train.py --help]
    """
    # Load Dataset.
    logging.info('loading LFW dataset into memory')
    X, IMAGE_SHAPE = load_dataset(dimx=36, dimy=36)

    tf.reset_default_graph()
    try:
        if not tf.test.is_gpu_available(cuda_only=True):
            raise Exception
    except Exception:
        logging.critical('CUDA capable GPU device not found.')
        exit(0)

    logging.warn('constructing graph on GPU')
    with tf.device('/gpu:0'):

        # Define placeholders for input data.
        noise = tf.placeholder('float32', [None, noise_dim])
        real_data = tf.placeholder('float32', [None, ] + list(IMAGE_SHAPE))

        # Create Generator and Discriminator models.
        logging.debug('creating generator and discriminator')
        g_out = generator(noise, train=True)
        d_probs, d_fake_logits = discriminator(g_out, train=True)
        d_probs2, d_real_logits = discriminator(real_data, train=True)

        logging.debug('defining training ops')
        # Define Generator(G) ops.
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))
        g_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr)
        g_vars = get_vars_by_scope('generator')
        g_train_step = g_optimizer.minimize(g_loss, var_list=g_vars)

        # Define Discriminator(D) ops.
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_real_logits)))
        d_loss = d_loss_real + d_loss_fake
        d_optimizer = tf.train.AdamOptimizer(learning_rate=disc_lr)
        d_vars = get_vars_by_scope('discriminator')
        d_train_step = d_optimizer.minimize(d_loss, var_list=d_vars)

    with tf.Session() as sess:
        # Init vars.
        sess.run(tf.global_variables_initializer())

        # Start training.
        logging.debug('training DCGAN model')
        for epoch in range(num_epochs):
            eval_noise = sample_noise_batch(16)
            idx = np.random.choice(range(X.shape[0]), size=16)
            eval_real_data = X[idx]
            for X_batch in tqdm(iterate_minibatches(X, batch_size, shuffle=True), 
                                total=X.shape[0] // batch_size, desc='Epoch[{}/{}]'
                                .format(epoch + 1, num_epochs), leave=False):
                sess.run([d_train_step], feed_dict={real_data: X_batch,
                                                    noise: sample_noise_batch(batch_size)})
                for _ in range(2):
                    sess.run([g_train_step], feed_dict={noise: sample_noise_batch(batch_size)})
            # Evaluating model after every epoch.
            d_loss_iter, g_loss_iter, eval_images = sess.run([d_loss, g_loss, g_out], 
                                                            feed_dict={real_data: eval_real_data,
                                                                    noise: eval_noise})
            # Generate images using G and save in `out/`.
            tl.visualize.save_images(eval_images, [4, 4], 'out/eval_{}.png'.format(epoch + 1))
            logging.info('Epoch[{}/{}]    g_loss: {:.6f}   -   d_loss: {:.6f}'
                        .format(epoch + 1, num_epochs, g_loss_iter, d_loss_iter))

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
    coloredlogs.install(level='DEBUG')
    try:
        train()
    except KeyboardInterrupt:
        logging.info('Aborted!')

if __name__ == '__main__':
    main()