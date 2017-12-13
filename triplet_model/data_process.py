import numpy as np
import os
import cv2
import utils

train_root = "/home/lenovo/yql/pig_data/train_folder_det_resize_no_pad"
valid_root = '/home/lenovo/yql/pig_data/valid_folder_det_resize_no_pad'
test_root = '/home/lenovo/yql/pig_data/test_B_resize_no_pad'

def one_hot(labels):
    labels = np.array(labels)
    one_hot = np.zeros((labels.shape[0], labels.max() + 1), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot

def read_data():
    train_data = []
    train_label = []
    for num, dirlist in enumerate(os.listdir(train_root)):
        label = int(dirlist[5:7]) - 1
        file_path = os.path.join(train_root, dirlist)
        train_data.append(file_path)
        train_label.append(label)
    np.savetxt('train_label.txt', train_label)

    valid_data = []
    valid_label = []
    name = []

    for dirlist in os.listdir(valid_root):
        label = int(dirlist[5:7]) - 1
        name.append(int(dirlist[8:16]))
        file_path = os.path.join(valid_root, dirlist)
        im = cv2.imread(file_path)
        width, heigh = im.shape[0], im.shape[1]
        if width >= heigh:
            length = heigh
        else:
            length = width
        print(length)
        im = utils.random_crop(im, image_size=length)
        im = cv2.resize(im, (128, 128))
        valid_data.append(im)
        valid_label.append(label)
    np.savetxt('valid_label.txt', valid_label)

    test_data = []
    for dirlist in os.listdir(test_root):
        file_path = os.path.join(test_root, dirlist)
        im = cv2.imread(file_path)
        test_data.append(im)

    return train_data, train_label, valid_data, valid_label, test_data, name

def input_test_data():
    test_data = []
    name = []
    for dirlist in os.listdir(test_root):
        im_name = str(dirlist.split('.')[0])
        name.append(im_name)
        file_path = os.path.join(test_root, dirlist)
        im = cv2.imread(file_path)
        width, heigh = im.shape[0], im.shape[1]
        if width >= heigh:
            length = heigh
        else:
            length = width
        print(length)
        im = utils.whiten(im)
        im = utils.random_crop(im, image_size=length)
        im = cv2.resize(im, (128, 128))
        test_data.append(im)
    print(name)
    print(len(test_data))
    print(len(name))
    np.savetxt('test_image_name.csv', name, fmt='%s')
    return np.array(test_data)

def input_data():
    train_data, train_label, _, _, _, name = read_data()

    return np.array(train_data), np.array(train_label)

def input_valid_data():
    _, _, valid_data, valid_label, _, name = read_data()
    # valid_label = one_hot(valid_label)
    np.random.seed(0)
    np.random.shuffle(valid_data)
    np.random.seed(0)
    np.random.shuffle(valid_label)
    np.random.seed(0)
    np.random.shuffle(name)
    np.savetxt('valid_image_name.csv', name, fmt='%s')

    print(np.shape(valid_data))
    print(np.shape(valid_label))


    return np.array(valid_data), np.array(valid_label)

# def input_path_dir():
#     train_path = []
#     train_label = []
#     train_no_det_path = []
#     for dir in os.listdir(train_root):
#         train_path.append(os.path.join(train_root, dir))
#         train_label.append(int(dir[5:7]) - 1)
#         train_no_det_path.append(os.path.join(train_no_det_root, dir))
#
#     train_label = one_hot(train_label)
#     np.random.seed(0)
#     np.random.shuffle(train_path)
#     np.random.seed(0)
#     np.random.shuffle(train_label)
#     np.random.seed(0)
#     np.random.shuffle(train_no_det_path)
#     return np.array(train_path), np.array(train_label), np.array(train_no_det_path)


def read_train_data():
    train_data = []
    train_label = []
    for num, dirlist in enumerate(os.listdir(train_root)):
        label = int(dirlist[5:7]) - 1
        file_path = os.path.join(train_root, dirlist)
        im = cv2.imread(file_path)
        width, heigh = im.shape[0], im.shape[1]
        if width >= heigh:
            length = heigh
        else:
            length = width
        # print(length)
        im = utils.whiten(im)
        im = utils.random_crop(im, image_size=length)
        im = cv2.resize(im, (128, 128))
        train_data.append(im)
        train_label.append(label)
    np.savetxt('train_label.txt', train_label)
    print("loading train_data, train_labels")
    return np.array(train_data), np.array(train_label)

if __name__ == '__main__':
    read_data()