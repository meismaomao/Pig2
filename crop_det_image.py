import tensorflow as tf
import numpy as np
import os
import cv2


train_root = '/home/lenovo/yql/pig_data/train_folder_det/'
save_root = '/home/lenovo/yql/pig_data/train_folder_det_random_crop/'


# print(image_raw_data)
image_raw_data = []
dir_list = []
for dir in os.listdir(train_root):
	train_path = os.path.join(train_root, dir)
	# print(train_path)
	image_raw_data.append(cv2.imread(train_path))
	dir_list.append(dir)
print(len(dir_list))

with tf.Session() as sess:
	print(dir_list)
	dir_list = dir_list[7000:]
	print(dir_list)
	for i, dir in enumerate(dir_list):
		img = image_raw_data[i+7000]
		num = np.shape(img)[0] if np.shape(img)[0] <= np.shape(img)[1] else np.shape(img)[1]
		print(num)
		img_placeholder = tf.placeholder(tf.float32)
		img_data = tf.random_crop(img_placeholder, (num, num, 3))
		img_data = tf.image.resize_images(img_data, (224, 224))

		img_data = sess.run(img_data, feed_dict={img_placeholder: img})
		save_path = os.path.join(save_root, dir)
		cv2.imwrite(save_path, img_data)
