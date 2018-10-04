import os
import click
import logging

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from model import create_generator, create_discriminator
from utils import load_dataset, sample_noise_batch, iterate_minibatches

emb_size = 256
g_lr = 0.001
d_lr = 0.004
batch_size = 256
num_epochs = 200

def train():

    X, IMAGE_SHAPE = load_dataset(dimx=36, dimy=36)

    noise = tf.placeholder('float32', [None, emb_size])
    real_data = tf.placeholder('float32', [None, ] + list(IMAGE_SHAPE))

    g_out, g_logits = create_generator(noise)
    d_out, d_fake = create_discriminator(g_out)
    d_out2, d_real = create_discriminator(real_data)

    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake))
    
    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake))
    d_loss = d_loss_real + d_loss_fake

    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lr)
    g_train_step = g_optimizer.minimize(g_loss)

    d_optimizer = tf.train.AdamOptimizer(learning_rate=d_lr)
    d_train_step = d_optimizer.minimize(d_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        for X_batch in tqdm(iterate_minibatches(X, batch_size, shuffle=True), 
                            total=(X.shape[0] // batch_size) + 1, 
                            desc='Epoch[{}/{}]'.format(epoch, num_epochs)):
            feed_dict = {
                real_data: X_batch,
                noise: sample_noise_batch(batch_size)
            }
            sess.run([d_train_step], feed_dict=feed_dict)
            sess.run([g_train_step], feed_dict=feed_dict)

    sess.close()

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO')
    try:
        train()
    except KeyboardInterrupt:
        logging.info('EXIT')

if __name__ == '__main__':
    main()