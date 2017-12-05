import os
from PIL import Image

image_root = r'C:\Users\Administrator\Desktop\JLBC_data_v1.0\cfw_60k'
train_save_root = r'C:\Users\Administrator\Desktop\JLBC_data_v1.0\train_image'
test_save_root = r'C:\Users\Administrator\Desktop\JLBC_data_v1.0\test_image'

for dir in os.listdir(image_root):
	dir1_path = os.path.join(image_root, dir)
	i = 40
	for dir2 in os.listdir(dir1_path):
		train_save_path = os.path.join(train_save_root, dir)
		test_save_path = os.path.join(test_save_root, dir)
		if not os.path.exists(train_save_path):
			os.mkdir(train_save_path)
		if not os.path.exists(test_save_path):
			os.mkdir(test_save_path)
		if i >= 0:
			im = Image.open(os.path.join(dir1_path, dir2))
			im.save(os.path.join(train_save_path, dir2), "JPEG")
			print(os.path.join(train_save_path, dir2))
			i = i - 1
		else:
			im = Image.open(os.path.join(dir1_path, dir2))
			im.save(os.path.join(test_save_path, dir2), "JPEG")
			print(os.path.join(test_save_path, dir2))
			
	