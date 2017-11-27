import numpy as np
import os
import cv2

train_root = "/home/lenovo/yql/pig_data/train_folder_det_resize"
valid_root = '/home/lenovo/yql/pig_data/validation_folder_det_resize'
test_root = '/home/lenovo/yql/pig_data/pig_test_resize'
train_no_det_root = '/home/lenovo/yql/pig_data/train_folder_resize'

def one_hot(labels):
    labels = np.array(labels)
    one_hot = np.zeros((labels.shape[0], labels.max() + 1), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot

def read_data():
    train_data = []
    train_label = []
    for dirlist in os.listdir(train_root):
        label = int(dirlist[5:7]) - 1
        file_path = os.path.join(train_root, dirlist)
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        train_data.append(im)
        train_label.append(label)

    valid_data = []
    valid_label = []
    for dirlist in os.listdir(valid_root):
        label = int(dirlist[5:7]) - 1
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

    return train_data, train_label, valid_data, valid_label, test_data

def input_test_data():
    test_data = []
    name = []
    for dirlist in os.listdir(test_root):
        im_name = str(dirlist.split('.')[0])
        name.append(im_name)
        file_path = os.path.join(test_root, dirlist)
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        test_data.append(im)
    print(name)
    np.savetxt('test_image_name.csv', name, fmt='%s')
    return test_data

def input_data():
    train_data, train_label, _, _, _ = read_data()
    train_label = one_hot(train_label)
    train_no_det_data, train_no_det_label = input_train_no_det()
    train_no_det_label = one_hot(train_no_det_label)

    np.random.seed(0)
    np.random.shuffle(train_data)
    np.random.seed(0)
    np.random.shuffle(train_label)
    np.random.seed(0)
    np.random.shuffle(train_no_det_data)
    np.random.seed(0)
    np.random.shuffle(train_no_det_label)

    return np.array(train_data), np.array(train_label) #, np.array(train_no_det_data), np.array(train_no_det_label)

def input_valid_data():
    _, _, valid_data, valid_label, _ = read_data()
    valid_label = one_hot(valid_label)

    np.random.seed(0)
    np.random.shuffle(valid_data)
    np.random.seed(0)
    np.random.shuffle(valid_label)

    print(np.shape(valid_data))
    print(np.shape(valid_label))


    return np.array(valid_data), np.array(valid_label)

def input_train_no_det():
    train_data = []
    train_label = []
    for dirlist in os.listdir(train_root):
        label = int(dirlist[5:7]) - 1
        file_path = os.path.join(train_no_det_root, dirlist)
        im = cv2.imread(file_path)
        # im = np.expand_dims(im, axis=0)
        train_data.append(im)
        train_label.append(label)
    return train_data, train_label

if __name__ == '__main__':
    valid_data, valid_label = input_valid_data()
    print(np.shape(valid_data))