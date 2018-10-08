""" Custom layer operations written in TensorFlow for DCGAN.

See https://arxiv.org/abs/1511.06434.pdf.
"""
import tensorflow as tf

def dense_layer(inputs, train, units, name, use_batchnorm=True, activation='leaky_relu'):
    """ Creates a dense layer with arbitrary number of hidden units with batch normalization and activation.

    Args:
        inputs (tf.placeholder): Inputs to the dense layer.
        train (bool): Flag for whether to freeze batch-norm layer vars.
        units (int): Number of hidden units in dense layer.
        name (str): Scope name for layer variables.
        use_batchnorm (bool, optional): Flag for whether to use batch-norm layer. If unspecified, defaults to `True`.
        activation (str, optional): Type of activation layer [relu/leaky_relu]. If unspecified, defaults to `leaky_relu`.
    Returns:
        Dense layer with given variable name.
    """
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

def conv_layer(
        inputs, 
        train, 
        kernel_dims, 
        in_channels, 
        out_channels, 
        name, 
        strides=(1, 1), 
        padding='SAME', 
        use_avgpool=True, 
        use_batchnorm=True, 
        activation='leaky_relu'
    ):
    """ Creates a convolutional layer with average pooling, batch normalization and activation.

    Args:
        inputs (tf.placeholder): Input tensor to the conv layer.
        train (bool): Flag for whether to freeze batch-norm layer vars.
        kernel_dims (tuple of `int`): Kernel dimensions (height and width) to use in conv op.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after conv op.
        name (str): Scope name for layer variables.
        strides (tuple of `int`, optional): Strides to use for conv op.
        padding (str, optional): Padding type to use for conv op. If unspecified, defaults to `SAME`.
        use_avgpool (bool, optional): Flag for whether to use average-pooling. If unspecified, defaults to `True`.
        use_batchnorm (bool, optional): Flag for whether to use batch-norm layer. If unspecified, defaults to `True`.
        activation (str, optional): Type of activation layer [relu/leaky_relu]. If unspecified, defaults to `leaky_relu`.
    Returns:
        Convolutional layer with given variable name.
    """
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('kernel', shape=list(kernel_dims) + [in_channels, out_channels], 
                                initializer=tf.glorot_uniform_initializer(), 
                                dtype=tf.float32)
        bias = tf.get_variable('bias', shape=[out_channels],
                                initializer=tf.constant_initializer(0.0), 
                                dtype=tf.float32)
        conv = tf.nn.conv2d(inputs, filter=kernel, 
                            strides=[1, strides[0], strides[1], 1], 
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

def deconv_layer(
        inputs, train, 
        kernel_dims, 
        in_channels, 
        out_channels, 
        batch_size, name, 
        strides=(1, 1), 
        padding='VALID', 
        use_batchnorm=True, 
        activation='leaky_relu'
    ):
    """ Creates a de-convolutional layer with batch normalization and activation.

    Args:
        inputs (tf.placeholder): Input tensor to the deconv layer.
        train (bool): Flag for whether to freeze batch-norm layer vars.
        kernel_dims (tuple of `int`): Kernel dimensions (height and width) to use in deconv op.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after deconv op.
        batch_size (tf.shape): Batch size to use when defining output shape in conv2d_transpose op.
        name (str): Scope name for layer variables.
        strides (tuple of `int`, optional): Strides to use for deconv op.
        padding (str, optional): Padding type to use for deconv op. If unspecified, defaults to `VALID`.
        use_batchnorm (bool, optional): Flag for whether to use batch-norm layer. If unspecified, defaults to `True`.
        activation (str, optional): Type of activation layer [relu/leaky_relu]. If unspecified, defaults to `leaky_relu`.
    Returns:
        De-convolutional layer with given variable name.
    """
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('kernel', shape=list(kernel_dims) + [out_channels, in_channels], 
                                initializer=tf.glorot_uniform_initializer(), dtype=tf.float32)
        bias = tf.get_variable('bias', shape=[out_channels],
                                initializer=tf.constant_initializer(0.0), 
                                dtype=tf.float32)
        input_dim = inputs.get_shape().as_list()[1]
        out_dim = input_dim * strides[0] + kernel_dims[0] - strides[0]
        deconv_shape = tf.stack([batch_size, out_dim, out_dim, out_channels])
        deconv = tf.nn.conv2d_transpose(inputs, filter=kernel, output_shape=deconv_shape,
                                        strides=[1, strides[0], strides[1], 1], padding=padding)
        deconv = tf.add(deconv, bias)
        if use_batchnorm:
            deconv = tf.layers.batch_normalization(deconv, training=train)
        if activation == 'relu':
            deconv = tf.nn.relu(deconv, name=scope.name)
        elif activation == 'leaky_relu':
            deconv = tf.nn.leaky_relu(deconv, alpha=0.2, name=scope.name)
    return deconv