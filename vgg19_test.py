import tensorflow as tf
import numpy as np
import utils
import data_process
import datetime

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
CLASS_COUNTS = 30
BATCH_SIZE = 30
MAX_ITRATIONS = int(1e10 + 1)
IMAGE_SIZE = 224
IMAGE_CHANNLES = 3
NUM = 3000
EPS = 1e-10



def subnet1(image, keep_prob, reuse=False):
	with tf.variable_scope("subnet1", reuse=reuse):
		def _vgg_net(weights, image):
			print('setting up vgg model initialized params --> extractor1')
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
					# kernels are [width, height, in_channles, out_channles]
					# tensorflow are [height, width, in channles, out_channles]
					kernels = utils.get_variable(
						np.transpose(kernels, (1, 0, 2, 3)), name=name + '_w')
					bias = utils.get_variable(bias.reshape(-1), name=name + '_b')
					current = utils.conv2d_basic(current, kernels, bias)
				elif kind == 'relu':
					current = tf.nn.relu(current, name=name)
				elif kind == 'pool':
					current = utils.avg_pool_2x2(current)
				net[name] = current
			return net

		model_data = utils.get_model_data("data", MODEL_URL)
		weights = np.squeeze(model_data['layers'])

		with tf.variable_scope('inference'):
			image_net = _vgg_net(weights, image)
			conv_final_layer = image_net['relu5_4']

			pool5 = utils.max_pool_2x2(conv_final_layer)

			pool5_flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
			W1 = utils.get_weights_variable(7 * 7 * 512, 4096, name='W1')
			b1 = utils.get_bias_variable(4096, name='b1')
			matmul1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool5_flatten, W1), b1))

			matmul1 = tf.nn.dropout(matmul1, keep_prob)

			W2 = utils.get_weights_variable(4096, 4096, name='W2')
			b2 = utils.get_bias_variable(4096, name='b2')
			matmul2 = tf.nn.bias_add(tf.matmul(matmul1, W2), b2)

			matmul2 = tf.nn.dropout(matmul2, keep_prob)

			W3 = utils.get_weights_variable(4096, CLASS_COUNTS, name='W3')
			b3 = utils.get_bias_variable(CLASS_COUNTS, name='b3')
			matmul3 = tf.nn.bias_add(tf.matmul(matmul2, W3), b3)

			return matmul3


def accuracy_fun(labels, logits):
	logits = tf.nn.softmax(logits)
	correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return accuracy

def main(argv=None):
	train_samples = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNLES], name='train_sample')
	train_labels = tf.placeholder(tf.float32, [None, CLASS_COUNTS], name='train_label')
	keep_prob = tf.placeholder(tf.float32, [], name='keep_probability')

	logits = subnet1(train_samples, keep_prob)
	softmax_logits = tf.nn.softmax(logits)

	test_data = data_process.input_test_data()

	Saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		Saver.restore(sess, './checkpoint/')
		print('Initialized!')
		prob = []
		for step in range(NUM//BATCH_SIZE):

			feed_dict = {train_samples: test_data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE], keep_prob: 1.0}

			softmax = sess.run(softmax_logits, feed_dict=feed_dict)

			prob.extend(softmax)

		np.savetxt('test_image_result.csv', prob, fmt='%g')
		for i in np.max(prob, 1):
			print(i)
		print(np.mean(np.max(prob, 1)))

if __name__ == '__main__':
	main()