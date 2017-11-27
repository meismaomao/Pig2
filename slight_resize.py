"""
this file function is use to resize the pig images,
we need to take into account the size ratio of the target object.
So this programme used to "slight resize" the pig images to same size, eg. 500x500.
Then, on this basis, we proceed with resize
"""

import os
import cv2
import numpy as np

SIZE = 500
scale = 40

TRAIN_DIR = '/home/lenovo/yql/pig_data/train_folder/'
SAVE_DIR = '/home/lenovo/yql/pig_data/train_folder_resize/'
# VALID_DIR = '/home/lenovo/yql/pig_data/train_folder_det/'
# file_name = 'image06-00000025.jpg'
for file_name in os.listdir(TRAIN_DIR):
	file_dir = os.path.join(TRAIN_DIR, file_name)
	print(file_dir)
	im = cv2.imread(file_dir)

	im_height = im.shape[0]
	im_width = im.shape[1]

	while(im_width > SIZE or im_height > SIZE):

		im_height = int(im_height / 1.15)
		im_width = int(im_width / 1.15)

		if im_width <= SIZE and im_height <= SIZE:
			im = cv2.resize(im, (im_width, im_height))
			break
		if np.abs(im_width - im_height) <= scale:
			im = cv2.resize(im, (SIZE, SIZE))
			break
		if im_width < SIZE and im_height - im_width < scale:
			im = cv2.resize(im, (im_height, SIZE))
			break
		if im_height < SIZE and im_width - im_height < scale:
			im = cv2.resize(im, (SIZE, im_height))
			break

	top = (SIZE - im.shape[1]) // 2 if (SIZE - im.shape[1]) % 2 == 0 else (SIZE - im.shape[1]) // 2 + 1
	bottom = (SIZE - im.shape[1]) // 2
	left = (SIZE - im.shape[0]) // 2 if (SIZE - im.shape[0]) % 2 == 0 else (SIZE - im.shape[0]) // 2 + 1
	right = (SIZE - im.shape[0]) // 2
	im = cv2.copyMakeBorder(im, left, right, top, bottom, cv2.BORDER_CONSTANT, value=[0, 0, 0])
	# cv2.imshow('img', im)
	# cv2.waitKey(25)
	##on this basis, we can resize the image and not change the ratio of width
	## and height. the pad does not influence the image classification
	im = cv2.resize(im, (224, 224))
	# class_name = file_name[5:7]
	# class_dir = os.path.join(SAVE_DIR, file_name)
	# if not os.path.exists(class_dir):
	# 	os.mkdir(class_dir)
	save_path = os.path.join(SAVE_DIR, file_name)
	cv2.imwrite(save_path, im)