import tensorflow as tf

def dense_layer(inputs, train, units, name, use_batchnorm=True, activation='leaky_relu'):
    with tf.variable_scope(name) as scope:
        input_dim = inputs.get_shape().as_list()[-1]
        weights = tf.get_variable('weights', shape=[input_dim, units], 
                                initializer=tf.glorot_uniform_initializer(), 
                                dtype=tf.float32)
        bias = tf.get_variable('bias', shape=[units], 
                                initializer=tf.constant_initializer(0.0), 
                                dtype=tf.float32)
        dense = tf.add(tf.matmul(inputs, weights), bias)
        if use_batchnorm:
            dense = tf.layers.batch_normalization(dense, training=train)
        if activation == 'relu':
            dense = tf.nn.relu(dense, name=scope.name)
        elif activation == 'leaky_relu':
            dense = tf.nn.leaky_relu(dense, alpha=0.2, name=scope.name)
    return dense

def conv_layer(inputs, train, kernel_dim, in_channels, out_channels, name, stride=1, 
            padding='SAME', use_avgpool=True, use_batchnorm=True, activation='leaky_relu'):
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('kernel', shape=[kernel_dim, kernel_dim, in_channels, out_channels], 
                                initializer=tf.glorot_uniform_initializer(), 
                                dtype=tf.float32)
        bias = tf.get_variable('bias', shape=[out_channels],
                                initializer=tf.constant_initializer(0.0), 
                                dtype=tf.float32)
        conv = tf.nn.conv2d(inputs, filter=kernel, 
                            strides=[1, stride, stride, 1], 
                            padding=padding)
        conv = tf.add(conv, bias)
        if use_batchnorm:
            conv = tf.layers.batch_normalization(conv, training=train)
        if activation == 'relu':
            conv = tf.nn.relu(conv)
        elif activation == 'leaky_relu':
            conv = tf.nn.leaky_relu(conv, alpha=0.2)
        if use_avgpool:
            conv = tf.layers.average_pooling2d(conv, pool_size=[2, 2], 
                                            strides=(1, 1), name=scope.name)
    return conv

def deconv_layer(inputs, train, kernel_dim, in_channels, out_channels, batch_size,
                    name, stride=1, use_batchnorm=True, activation='leaky_relu'):
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('kernel', shape=[kernel_dim, kernel_dim, out_channels, in_channels], 
                                initializer=tf.glorot_uniform_initializer(), dtype=tf.float32)
        bias = tf.get_variable('bias', shape=[out_channels],
                                initializer=tf.constant_initializer(0.0), 
                                dtype=tf.float32)
        input_dim = inputs.get_shape().as_list()[1]
        out_dim = input_dim * stride + kernel_dim - stride
        deconv_shape = tf.stack([batch_size, out_dim, out_dim, out_channels])
        deconv = tf.nn.conv2d_transpose(inputs, filter=kernel, output_shape=deconv_shape,
                                        strides=[1, stride, stride, 1], padding='VALID')
        deconv = tf.add(deconv, bias)
        if use_batchnorm:
            deconv = tf.layers.batch_normalization(deconv, training=train)
        if activation == 'relu':
            deconv = tf.nn.relu(deconv, name=scope.name)
        elif activation == 'leaky_relu':
            deconv = tf.nn.leaky_relu(deconv, alpha=0.2, name=scope.name)
    return deconv