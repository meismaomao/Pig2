import tensorflow as tf
import numpy as np
import utils
import squeeze_net
import os
import importlib
import data_process
from DataShuffle import DataShuffle
import inception_resnet_v1
import datetime

def main(argv=None):

	with tf.Graph(). as_default():

		BATCH_SIZE = 10
		NUM = 860

		train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 128, 128, 3], name='anchor')

		labels_anchor = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 30])

		_, prelogits, _ = inception_resnet_v1.inference(train_anchor_data, keep_probability=1.0,
		                                 phase_train=False, bottleneck_layer_size=30,
		                                 weight_decay=0.0005)

		clas_loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=labels_anchor, logits=prelogits))
		logits = tf.nn.softmax(prelogits)
		clas_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels_anchor, 1), tf.argmax(logits, 1)), tf.float32),
		                           name='clas_acc')
		Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

		data, labels = data_process.input_valid_data()
		print(data.shape)
		print(labels.shape)

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			Saver.restore(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard2/')
			print('Initialized!')
			loss = 0
			accuracy = 0
			for step in range(NUM // BATCH_SIZE):
				batch_anchor, batch_labels_anchor = data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],\
				                                    labels[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

				feed_dict = {train_anchor_data: batch_anchor, labels_anchor: batch_labels_anchor}
				Clas_loss, Clas_acc = sess.run([clas_loss, clas_acc], feed_dict=feed_dict)

				print("%d the %s train reslut" % (step, datetime.datetime.now()))
				print('the Clas_loss %g' % Clas_loss)
				print('the Clas_acc %g' % Clas_acc)
				loss += Clas_loss
				accuracy += Clas_acc
			loss = loss / float(NUM // BATCH_SIZE)
			accuracy = accuracy / float(NUM // BATCH_SIZE)
			print(loss)
		print(accuracy)

if __name__ == '__main__':
	main()