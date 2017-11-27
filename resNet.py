import tensorflow as tf
import numpy as np

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))
    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)
    return fc_h

def relu_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))
    fc_h = tf.nn.relu(tf.matmul(inpt, fc_w) + fc_b)
    return fc_h

def conv_layer(inpt, filter_shape, stride):
    out_channles = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    bias_ = tf.Variable(tf.zeros([out_channles]))
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, bias_)
    mean, var = tf.nn.moments(conv, axis=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channles]), name='beta')
    gamma = weight_variable([out_channles], name='gamma')

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.0001, scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)
    return out

def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()
    if down_sample:
        filter_ = [1, 2, 2, 1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=[1, 2, 2, 1])

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            stride = 2 if down_sample else 1
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], stride)
        else:
            input_layer = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer

    return res
