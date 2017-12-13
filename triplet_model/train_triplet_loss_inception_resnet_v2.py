import tensorflow as tf
import utils
import data_process
from DataShuffle import DataShuffle
import inception_resnet_v1
import datetime

model_def = '/home/lenovo/yql/pig_data/model_save/'

def main(argv=None):

	with tf.Graph(). as_default():

		global_step = tf.Variable(0, trainable=False)
		CENTER_LOSS_ALPHA = 1.25
		BATCH_SIZE = 256
		ITERATION = 2000000
		data_num = 16640
		NUM_CLASSES = 30
		train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 128, 128, 3], name='anchor')

		labels_anchor = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
		tf.summary.image('input_image', train_anchor_data, 10)

		features, prelogits, logits = inception_resnet_v1.inference(train_anchor_data, keep_probability=0.8,
		                                 phase_train=True, bottleneck_layer_size=30,
		                                 weight_decay=0.0005)
		print(features, prelogits, logits)
		with tf.name_scope('loss'):
			with tf.name_scope('center_loss'):
				center_loss, centers, centers_update_op = utils.get_center_loss(features, labels_anchor, CENTER_LOSS_ALPHA, NUM_CLASSES)
			with tf.name_scope('softmax_loss'):
				softmax_loss = tf.reduce_mean(
					tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_anchor, logits=prelogits))
			with tf.name_scope('total_loss'):
				total_loss = softmax_loss + center_loss

		# global_step = tf.Variable(tf.constant(431, dtype=tf.int64))
		lr = tf.train.exponential_decay(
			learning_rate=0.001,
			global_step=global_step,
			decay_steps=data_num // BATCH_SIZE,
			decay_rate=0.99,
			staircase=True
		)
		with tf.control_dependencies([centers_update_op]):
			train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, global_step=global_step)
		Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

		tf.summary.scalar('total_loss', total_loss)
		tf.summary.scalar('learning_rate', lr)
		tf.summary.scalar('softmax_loss', softmax_loss)
		tf.summary.scalar('center_loss', center_loss)
		merged = tf.summary.merge_all()

		data, labels = data_process.input_data()
		print(data.shape)
		print(labels.shape)
		dataShufflu = DataShuffle(data, labels)
		with tf.Session() as sess:
			train_write = tf.summary.FileWriter('/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard4/',
			                                    sess.graph)

			sess.run(tf.global_variables_initializer())
			# sess.run(tf.local_variables_initializer())
			# Saver.restore(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard2/')

			for step in range(ITERATION):
				batch_anchor, batch_labels_anchor = dataShufflu.get_triplet(n_labels=30, n_triplet=BATCH_SIZE)

				feed_dict = {
					train_anchor_data: batch_anchor,
					labels_anchor: batch_labels_anchor
				}

				_, l, summary, Loss = sess.run([train_op, total_loss, merged, softmax_loss], feed_dict=feed_dict)

				print("%d the %s train reslut" % (step, datetime.datetime.now()))
				print('the softmax loss %g' % Loss)
				print('the total loss %g' % l)

				train_write.add_summary(summary, step)

				if step % 500 == 0:
					Saver.save(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard4/')

			train_write.close()

if __name__ == '__main__':
	main()