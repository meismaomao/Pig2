import tensorflow as tf
import numpy as np
import utils
import data_process
from DataShuffle import DataShuffle
import inception_resnet_v1
import datetime
import pickle
from sklearn.svm import SVC

def main(data, iter_num, labels):

	with tf.Graph(). as_default():

		BATCH_SIZE = 866
		train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 128, 128, 3], name='anchor')

		labels_anchor = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

		features, prelogits, logits = inception_resnet_v1.inference(train_anchor_data, keep_probability=1.0,
		                                 phase_train=False, bottleneck_layer_size=30,
		                                 weight_decay=0.0005)
		features = tf.nn.l2_normalize(features, 1, 1e-10, name='embeddings')

		data_train, train_labels = data, labels
		print(data.shape)
		print(labels.shape)
		Saver = tf.train.Saver()
		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			emb_array = np.zeros((iter_num // BATCH_SIZE * BATCH_SIZE, 1792))
			for step in range(iter_num // BATCH_SIZE):
				batch_anchor, batch_labels = data_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE], \
				                             train_labels[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
				# batch_anchor = data_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
				Saver.restore(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard5/')
				feed_dict = {
					train_anchor_data: batch_anchor,
					labels_anchor: batch_labels,
				}

				Logits = sess.run(features, feed_dict=feed_dict)
				emb_array[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, :] = Logits
				print('the triplet loss %g' % step)
		np.savetxt('valid_feature_12_11.txt', emb_array)
		print(np.shape(emb_array))
		# if train:
		# 	print('Training classifier')
		# 	class_name = ['%2d' % (i) for i in range(CLASSES_NUM)]
		# 	print(class_name)
		# 	SVMmodel = SVC(kernel='rbf', probability=True)
		# 	SVMmodel.fit(emb_array, labels[:ITERATION // BATCH_SIZE * BATCH_SIZE])
		# 	with open('train_model_pickle30', 'wb') as outfile:
		# 		pickle.dump((SVMmodel, class_name), outfile)
		# 	print('Saved classifier model to file train_model_pickle')
		# else:
		# 	print('Testing classifier30')
		# 	with open('train_model_pickle30', 'rb') as infile:
		# 		(knn, class_name) = pickle.load(infile)
		# 	print('Loaded classifier model from file train_model_pickle')
		#
		# 	predictions = knn.predict_proba(emb_array)
		# 	np.savetxt('test_k_30_image_result.csv', predictions, fmt='%g')
		# 	best_class_indices = np.argmax(predictions, axis=1)
		# 	best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
		#
		# 	for i in range(len(best_class_indices)):
		# 		print('%4d %s %0.3f' % (i, class_name[best_class_indices[i]], best_class_probabilities[i]))
		# 	accuracy = np.mean(np.equal(best_class_indices, labels))
		# 	print("Accuracy %.3f" % accuracy)

if __name__ == "__main__":
	# data, label = data_process.read_train_data()
	# num = data.shape[0]
	# print(np.shape(data))
	# print(np.shape(label))
	# main(data=data, iter_num=num, labels=label)
	data, label = data_process.input_valid_data()
	main(data=data, iter_num=866, labels=label)
	# data = data_process.input_test_data()
	# main(data=data, iter_num=3000, labels=None, train=False)