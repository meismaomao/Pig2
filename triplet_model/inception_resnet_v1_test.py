import tensorflow as tf
import numpy as np
import data_process
import inception_resnet_v1
import datetime

def main(argv=None):

	with tf.Graph(). as_default():

		BATCH_SIZE = 3000
		NUM = 3000

		train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 128, 128, 3], name='anchor')

		_, prelogits, _ = inception_resnet_v1.inference(train_anchor_data, keep_probability=1.0,
		                                 phase_train=False, bottleneck_layer_size=30,
		                                 weight_decay=0.0005)

		logits = tf.nn.softmax(prelogits)

		Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

		data = data_process.input_test_data()
		print(data.shape)

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			Saver.restore(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard5/')
			print('Initialized!')
			prob = []
			for step in range(NUM // BATCH_SIZE):
				batch_anchor = data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

				feed_dict = {train_anchor_data: batch_anchor}
				Logits = sess.run(logits, feed_dict=feed_dict)

				print("%d the %s train reslut" % (step, datetime.datetime.now()))
				print('the i %g' % step)
				prob.extend(Logits)
		print(np.shape(prob))
		np.savetxt('test_image_result.csv', prob, fmt='%g')
		for i in np.max(prob, 1):
			print(i)
	print(np.mean(np.max(prob, 1)))

if __name__ == '__main__':
	main()