import tensorflow as tf
import numpy as np
import datetime
import utils
import data_process


MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
CLASS_COUNTS = 256

def NetworkModel(image, keep_prob, reuse=False):
	with tf.variable_scope('NetworkModel', reuse=reuse):
		def _vgg_net(weights, image):
			layers = (
				'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                'relu5_3', 'conv5_4', 'relu5_4'
			)
			net = {}
			current = image
			for i, name in enumerate(layers):
				kind = name[:4]
				if kind == 'conv':
					kernels, bias = weights[i][0][0][0][0]
					kernels = utils.get_variable(
						np.transpose(kernels, (1, 0, 2, 3)), name=name + '_w')
					bias = utils.get_variable(bias.reshape(-1), name=name + '_b')
					current = utils.conv2d_basic(current, kernels, bias)
					tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0005)(kernels))
				elif kind == 'relu':
					current = tf.nn.relu(current, name=name)
				elif kind == 'pool':
					current = utils.max_pool_2x2(current)
				net[name] = current
			return net
		print('setting up vgg model initialized params --> extractor')
		model_data = utils.get_model_data("data", MODEL_URL)
		weights = np.squeeze(model_data['layers'])

		with tf.name_scope('inference'):
			image_net = _vgg_net(weights, image)
			conv_final_layer = image_net['relu5_4']

			pool5 = utils.max_pool_2x2(conv_final_layer)

			pool5_flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
			W1 = utils.get_weights_variable(7 * 7 * 512, 4096, name='W1')
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0005)(W1))
			b1 = utils.get_bias_variable(4096, name='b1')
			matmul1 = tf.nn.bias_add(tf.matmul(pool5_flatten, W1), b1)
			matmul1 = tf.nn.relu(matmul1)

			matmul1 = tf.nn.dropout(matmul1, keep_prob)

			W2 = utils.get_weights_variable(4096, 4096, name='W2')
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0005)(W2))
			b2 = utils.get_bias_variable(4096, name='b2')
			matmul2 = tf.nn.bias_add(tf.matmul(matmul1, W2), b2)
			matmul2 = tf.nn.relu(matmul2)

			matmul2 = tf.nn.dropout(matmul2, keep_prob)

			W3 = utils.get_weights_variable(4096, CLASS_COUNTS, name='W3')
			tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0005)(W3))
			b3 = utils.get_bias_variable(CLASS_COUNTS, name='b3')
			pre_logits = tf.nn.bias_add(tf.matmul(matmul2, W3), b3)
			# logits = tf.nn.softmax(pre_logits)
			# regularizer_losses = tf.get_collection("losses")
			return pre_logits   #, logits

