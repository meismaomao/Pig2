import numpy as np
import os
import cv2
import utils

train_root = "/home/lenovo/yql/pig_data/train_folder_det_resize"
valid_root = '/home/lenovo/yql/pig_data/validation_folder_det_resize'
test_root = '/home/lenovo/yql/pig_data/pig_test_resize'

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
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        im = 2 * (im / 255.0) - 1.0
        train_data.append(im)
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
        # im = np.expand_dims(im, axis=0)
        valid_data.append(im)
        valid_label.append(label)

    test_data = []
    for dirlist in os.listdir(test_root):
        file_path = os.path.join(test_root, dirlist)
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        test_data.append(im)

    return train_data, train_label, valid_data, valid_label, test_data, name

def input_test_data():
    test_data = []
    test_no_det_data = []
    name = []
    for dirlist in os.listdir(test_root):
        im_name = str(dirlist.split('.')[0])
        name.append(im_name)
        file_path = os.path.join(test_root, dirlist)
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        test_data.append(im)


    print(name)
    #print(len(test_no_det_data))
    print(len(test_data))
    print(len(name))
    np.savetxt('test_image_name.csv', name, fmt='%s')
    return np.array(test_data)  #, np.array(test_no_det_data)

def input_data():
    train_data, train_label, _, _, _, name = read_data()
    # train_label = one_hot(train_label)
    # train_no_det_data, train_no_det_label = input_train_no_det()
    # train_no_det_label = one_hot(train_no_det_label)

    np.random.seed(0)
    np.random.shuffle(train_data)
    np.random.seed(0)
    np.random.shuffle(train_label)

    return np.array(train_data), np.array(train_label) #, np.array(train_no_det_data), np.array(train_no_det_label)

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

def read_input_train_data(dir_list):
    train_data = []
    for file_path in dir_list:
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        train_data.append(im)
    return np.array(train_data)

def read_input_train_no_det_data(dir_list):
    train_data = []
    for file_path in dir_list:
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        train_data.append(im)
    return np.array(train_data)

if __name__ == '__main__':
    read_data()