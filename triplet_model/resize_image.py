import cv2
import numpy as np
import os

train_root = '/home/lenovo/yql/pig_data/train_folder_det_resize'
i = 1
for file1 in os.listdir(train_root):
	read_path = os.path.join(train_root, file1)
	print(read_path)
	im = cv2.imread(read_path)
	im = cv2.resize(im, (int(299 * 1.1), int(299 * 1.1)))
	cv2.imwrite(read_path, im)
	i = i + 1
