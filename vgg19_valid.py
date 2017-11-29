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
NUM = 840
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
			matmul1 = tf.nn.bias_add(tf.matmul(pool5_flatten, W1), b1)
			matmul1_out = tf.nn.relu(matmul1)

			matmul1 = tf.nn.dropout(matmul1_out, keep_prob)

			W2 = utils.get_weights_variable(4096, 4096, name='W2')
			b2 = utils.get_bias_variable(4096, name='b2')
			matmul2 = tf.nn.bias_add(tf.matmul(matmul1, W2), b2)
			matmul2 = tf.nn.relu(matmul2)

			matmul2 = tf.nn.dropout(matmul2, keep_prob)

			W3 = utils.get_weights_variable(4096, CLASS_COUNTS, name='W3')
			b3 = utils.get_bias_variable(CLASS_COUNTS, name='b3')
			matmul3 = tf.nn.bias_add(tf.matmul(matmul2, W3), b3)

			return matmul3, matmul1_out

def subnet2(image, keep_prob, reuse=False):
	with tf.variable_scope("subnet2", reuse=reuse):
		def _vgg_net(weights, image):
			print('setting up vgg model initialized params --> extractor2')
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
			matmul1 = tf.nn.bias_add(tf.matmul(pool5_flatten, W1), b1)
			matmul1_out = tf.nn.relu(matmul1)

			matmul1 = tf.nn.dropout(matmul1_out, keep_prob)

			W2 = utils.get_weights_variable(4096, 4096, name='W2')
			b2 = utils.get_bias_variable(4096, name='b2')
			matmul2 = tf.nn.bias_add(tf.matmul(matmul1, W2), b2)
			matmul2 = tf.nn.relu(matmul2)

			matmul2 = tf.nn.dropout(matmul2, keep_prob)

			W3 = utils.get_weights_variable(4096, CLASS_COUNTS, name='W3')
			b3 = utils.get_bias_variable(CLASS_COUNTS, name='b3')
			matmul3 = tf.nn.bias_add(tf.matmul(matmul2, W3), b3)

			return matmul3, matmul1_out

def all_classifier(feature1, feature2, keep_prob):
	with tf.name_scope('all_classifier'):
		W1 = utils.get_weights_variable(4096, 3072, name='W1')
		b1 = utils.get_bias_variable(3072, name='b1')
		matmul1 = tf.nn.bias_add(tf.matmul(feature1, W1), b1)
		matmul1 = tf.nn.relu(matmul1)

		W2 = utils.get_weights_variable(4096, 1024, name='W2')
		b2 = utils.get_bias_variable(1024, name='b2')
		matmul2 = tf.nn.bias_add(tf.matmul(feature2, W2), b2)
		matmul2 = tf.nn.relu(matmul2)

		feature = tf.concat([matmul1, matmul2], axis=1)

		feature = tf.nn.dropout(feature, keep_prob)

		W3 = utils.get_weights_variable(4096, 3072, name='W3')
		b3 = utils.get_bias_variable(3072, name='b3')
		matmul3 = tf.nn.bias_add(tf.matmul(feature, W3), b3)
		matmul3 = tf.nn.relu(matmul3)

		matmul3 = tf.concat([matmul3, matmul2], axis=1)
		matmul3 = tf.nn.dropout(matmul3, keep_prob)

		W4 = utils.get_weights_variable(4096, CLASS_COUNTS, name='W4')
		b4 = utils.get_bias_variable(CLASS_COUNTS, name='b4')
		matmul4 = tf.nn.bias_add(tf.matmul(matmul3, W4), b4)

		return matmul4



def accuracy_fun(labels, logits):
	logits = tf.nn.softmax(logits)
	correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return accuracy


def main(argv=None):
	train_samples = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNLES], name='train_sample')
	train_labels = tf.placeholder(tf.float32, [None, CLASS_COUNTS], name='train_label')
	keep_prob = tf.placeholder(tf.float32, [], name='keep_probability')

	logits1, feature_out1 = subnet1(train_samples, keep_prob)
	logits2, feature_out2 = subnet2(train_samples, keep_prob)
	logits3 = all_classifier(feature_out1, feature_out2, keep_prob)

	clas_loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits3))
	clas3_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_labels, 1), tf.argmax(logits3, 1)), tf.float32),
	                           name='clas3_acc')

	valid_data, valid_label = data_process.input_valid_data()

	Saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		Saver.restore(sess, './checkpoint_dual_vgg19/')
		print('Initialized!')
		loss = 0
		accuracy = 0
		for step in range(NUM//BATCH_SIZE):

			feed_dict = {train_samples: valid_data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE], train_labels:
				valid_label[step * BATCH_SIZE:(step + 1) * BATCH_SIZE], keep_prob: 1.0}

			Clas1_loss = sess.run(clas_loss, feed_dict=feed_dict)

			Clas1_acc = sess.run(clas3_acc, feed_dict=feed_dict)
			print('the classifier one loss %g' % Clas1_loss)
			print('the classifier1 accuracy is %g' % (Clas1_acc))

			loss += Clas1_loss
			accuracy += Clas1_acc
		loss = loss/float(NUM//BATCH_SIZE)
		accuracy = accuracy / float(NUM // BATCH_SIZE)
		print(loss)
		print(accuracy)

if __name__ == '__main__':
	main()