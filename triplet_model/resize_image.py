import cv2
import numpy as np
import os

train_root = '/home/lenovo/yql/pig_data/test_B'
save_root = '/home/lenovo/yql/pig_data/test_B_resize_no_pad'

i = 1

for file1 in os.listdir(train_root):
	read_path = os.path.join(train_root, file1)
	print(read_path)
	im = cv2.imread(read_path)
	im = np.array(im)
	print(im.shape)
	save_path = os.path.join(save_root, file1)
	if np.abs(im.shape[1] - im.shape[0]) <= 20:
		im = cv2.resize(im, (int(299 * 1.05), int(299 * 1.05)))
		print(im.shape)
		cv2.imwrite(save_path, im)
		i = i + 1
		continue
	if im.shape[0] - im.shape[1] > 0:
		im = cv2.resize(im, (int(299 * 1.05), int(299 * 1.05 * im.shape[0] / im.shape[1])))
		print(im.shape)
		cv2.imwrite(save_path, im)
		i = i + 1
		continue
	if im.shape[0] - im.shape[1] < 0:
		im = cv2.resize(im, (int(299 * 1.05 * im.shape[1] / im.shape[0]), int(299 * 1.05)))
		print(im.shape)
		cv2.imwrite(save_path, im)
		i = i + 1
		continue
	print(save_path)
