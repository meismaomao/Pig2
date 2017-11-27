import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block, relu_layer

def resnet(inpt, num_conv, batch_size, keep_prob):
    layers = []

    with tf.name_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 64], 1)
        layers.append(conv1)

    for i in range(num_conv):
        with tf.name_scope('conv2_%d' % (i + 1)):
            conv2_x = residual_block(layers[-1], 64, False)
            conv2 = residual_block(conv2_x, 64, False)
            layers.append(conv2_x)
            layers.append(conv2)

            assert conv2.get_shape().as_list()[1:] == [224, 224, 64]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.name_scope('conv3_%d' % (i + 1)):
            conv3_x = residual_block(layers[-1], 128, down_sample)
            conv3 = residual_block(conv3_x, 128, False)
            layers.append(conv3_x)
            layers.append(conv3)

            assert conv3.get_shape().as_list()[1:] == [112, 112, 128]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.name_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 256, down_sample)
            conv4 = residual_block(conv4_x, 256, False)
            layers.append(conv4_x)
            layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [56, 56, 256]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.name_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 512, down_sample)
            conv4 = residual_block(conv4_x, 512, False)
            layers.append(conv4_x)
            layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [28, 28, 512]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.name_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 512, down_sample)
            conv4 = residual_block(conv4_x, 512, False)
            layers.append(conv4_x)
            layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [14, 14, 512]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.name_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 512, down_sample)
            conv4 = residual_block(conv4_x, 512, False)
            layers.append(conv4_x)
            layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [7, 7, 512]


    with tf.name_scope('fc'):
        # global_pool = tf.reduce_mean(layers[-1], [1, 2])
        global_conv = conv_layer(inpt, [7, 7, 512, 4096], 1)
        print(global_conv.get_shape().as_list())
        global_conv_flatten = tf.reshape(global_conv, [batch_size, -1])

        fc1 = relu_layer(global_conv_flatten, (4096, 4096))

        fc1 = tf.nn.dropout(fc1, keep_prob)

        fc2 = relu_layer(fc1, (4096, 4096))

        out = softmax_layer(fc2, [4096, 30])
        layers.append(out)

    return layers[-1]

