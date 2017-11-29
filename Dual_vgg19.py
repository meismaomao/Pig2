import tensorflow as tf
import numpy as np
import utils
import data_process
import datetime

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
CLASS_COUNTS = 30
BATCH_SIZE = 64
MAX_ITRATIONS = int(1e10 + 1)
IMAGE_SIZE = 224
IMAGE_CHANNLES = 3
RAW_NUM = 14728
EPOCH = 10
EPOCH1 = 5
TRAIN_NUM = 14728
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
			matmul1_out = tf.nn.bias_add(tf.matmul(pool5_flatten, W1), b1)
			matmul1 = tf.nn.relu(matmul1_out)

			matmul1 = tf.nn.dropout(matmul1, keep_prob)

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
			matmul1_out = tf.nn.bias_add(tf.matmul(pool5_flatten, W1), b1)
			matmul1 = tf.nn.relu(matmul1_out)

			matmul1 = tf.nn.dropout(matmul1, keep_prob)

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
		W1 = utils.get_weights_variable(4096, 2048, name='W1')
		b1 = utils.get_bias_variable(2048, name='b1')
		matmul1 = tf.nn.bias_add(tf.matmul(feature1, W1), b1)
		matmul1 = tf.nn.relu(matmul1)

		W2 = utils.get_weights_variable(4096, 2048, name='W2')
		b2 = utils.get_bias_variable(2048, name='b2')
		matmul2 = tf.nn.bias_add(tf.matmul(feature2, W2), b2)
		matmul2 = tf.nn.relu(matmul2)

		feature = tf.concat([matmul1, matmul2], axis=1)

		feature = tf.nn.dropout(feature, keep_prob)

		W3 = utils.get_weights_variable(4096, 4096, name='W3')
		b3 = utils.get_bias_variable(4096, name='b3')
		matmul3 = tf.nn.bias_add(tf.matmul(feature, W3), b3)
		matmul3 = tf.nn.relu(matmul3)

		W5 = utils.get_weights_variable(4096, CLASS_COUNTS, name='W5')
		b5 = utils.get_bias_variable(CLASS_COUNTS, name='b5')
		matmul5 = tf.nn.bias_add(tf.matmul(matmul3, W5), b5)

		return matmul5


def main(argv=None):
	train_samples = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNLES], name='train_sample')
	train_labels = tf.placeholder(tf.float32, [None, CLASS_COUNTS], name='train_label')
	train_no_det_samples = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNLES], name='train_sample1')
	keep_prob = tf.placeholder(tf.float32, [], name='keep_probability')

	logits1, feature_out1 = subnet1(train_samples, keep_prob)
	logits2, feature_out2 = subnet2(train_no_det_samples, keep_prob)
	logits4 = all_classifier(feature_out1, feature_out2, keep_prob)

	# clas1_loss = tf.reduce_mean(
	# 	tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits1))
	# clas2_loss = tf.reduce_mean(
	# 	tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits2))
	clas4_loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits4))

	# clas1_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_labels, 1), tf.argmax(logits1, 1)), tf.float32),
	#                            name='clas1_acc')
	# clas2_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_labels, 1), tf.argmax(logits2, 1)), tf.float32),
	#                            name='clas2_acc')
	clas4_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_labels, 1), tf.argmax(logits4, 1)), tf.float32),
	                           name='clas4_acc')

	# classifier1_op = tf.train.AdamOptimizer(0.0005).minimize(classifier1_loss)
	# var = [var for var in tf.trainable_variables() if var.name.startswith('all_classifier')]
	# classifier1_op = tf.train.GradientDescentOptimizer(0.0005).minimize(clas1_loss)
	# classifier2_op = tf.train.GradientDescentOptimizer(0.0005).minimize(clas2_loss)
	classifier4_op = tf.train.GradientDescentOptimizer(0.0005).minimize(clas4_loss)

	train_path, train_label, train_no_det_path = data_process.read_train_det_or_no_det()

	Saver = tf.train.Saver()
	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter('./checkpoint_triple_vgg19/', sess.graph)
		sess.run(tf.global_variables_initializer())
		# Saver.restore(sess, './checkpoint_dual_vgg19/')
		# print('Initialized!')
		# for step in range(int(5000)):
		#
		# 	train_offset = (step * BATCH_SIZE) % (TRAIN_NUM - BATCH_SIZE)
		# 	batch_data = train_path[train_offset:(train_offset + BATCH_SIZE)]
		# 	batch_label = train_label[train_offset:(train_offset + BATCH_SIZE), :]
		# 	batch_data_no_det = train_no_det_path[train_offset:(train_offset + BATCH_SIZE)]
		#
		# 	feed_dict = {train_samples: batch_data, train_labels: batch_label,
		# 	             train_no_det_samples: batch_data_no_det, keep_prob: 0.80}
		#
		# 	_, Clas1_loss = sess.run([classifier1_op, clas1_loss], feed_dict=feed_dict)
		# 	_, Clas2_loss = sess.run([classifier2_op, clas2_loss], feed_dict=feed_dict)
		#
		# 	Clas1_acc = sess.run(clas1_acc, feed_dict=feed_dict)
		# 	Clas2_acc = sess.run(clas2_acc, feed_dict=feed_dict)
		#
		# 	print("%d the %s train reslut" % (step, datetime.datetime.now()))
		# 	print('the classifier one loss %g' % Clas1_loss)
		# 	print('the classifier1 accuracy is %g' % Clas1_acc)
		# 	print('the classifier two loss %g' % Clas2_loss)
		# 	print('the classifier2 accuracy is %g' % Clas2_acc)
		#
		# 	# if step % 1000 == 0:
		# 	# 	Saver.save(sess, './checkpoint_dual_vgg19/')
		#
		# print('Initialized train all clasifier!!')
		for step in range(int(5000 + 1)):
			train_offset = (step * BATCH_SIZE) % (TRAIN_NUM - BATCH_SIZE)
			batch_data = train_path[train_offset:(train_offset + BATCH_SIZE)]
			batch_label = train_label[train_offset:(train_offset + BATCH_SIZE), :]

			batch_data_no_det = train_no_det_path[train_offset:(train_offset + BATCH_SIZE)]

			feed_dict = {train_samples: batch_data, train_labels: batch_label,
			             train_no_det_samples: batch_data_no_det, keep_prob: 0.80}

			_, Clas4_loss = sess.run([classifier4_op, clas4_loss], feed_dict=feed_dict)

			Clas4_acc = sess.run(clas4_acc, feed_dict=feed_dict)

			print("%d the %s train reslut" % (step, datetime.datetime.now()))
			print('the classifier loss %g' % Clas4_loss)
			print('the classifier accuracy is %g' % Clas4_acc)

			if step % 100 == 0:
				Saver.save(sess, './checkpoint_triple_vgg19/')

if __name__ == '__main__':
	main()