import tensorflow as tf
import numpy as np
import os


train_root = '/home/lenovo/yql/pig_data/pig_test/'
save_root = '/home/lenovo/yql/pig_data/test_folder_random_crop/'

for dir in os.listdir(train_root):
	train_path = os.path.join(train_root, dir)
	# print(train_path)
	image_raw_data = tf.gfile.FastGFile(train_path, 'rb').read()
	# print(image_raw_data)

	with tf.Session() as sess:
		image_data = tf.image.decode_jpeg(image_raw_data)
		im = image_data.eval()
		num = np.shape(im)[0] if np.shape(im)[0] <= np.shape(im)[1] else np.shape(im)[1]
		print(num)
		image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

		img_data = tf.random_crop(image_data, (num, num, 3))
		img_data = tf.image.resize_images(img_data, (224, 224))

		img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
		encode_image = tf.image.encode_jpeg(img_data)
		save_path = os.path.join(save_root, dir)
		with tf.gfile.GFile(save_path, 'wb') as f:
			f.write(encode_image.eval())
