# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, sys
import scipy.misc as misc
import tarfile
import zipfile
import scipy.io
import urllib
import cv2
import random
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

def maybe_download_and_extract(dir_path, model_url, is_zipfile=False, is_tarfile=False):
    """
    Modified implementation from tensorflow/model/cifar10/input_data
    :param dir_path:
    :param model_url:
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = model_url.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Download %s %.1f%%' %(filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        file_path, _ = urllib.urlretrieve(model_url, filepath, reporthook=_progress)
        print('\n')
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                # zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)
        elif is_tarfile:
            tarfile.open(file_path, 'r:gz').extractall(dir_path)

# def get_vgg19_model_params(dir_path, model_url):
def save_image_pre_annotation(pre_annotation, train_image, annotation):
    pre_annotation = np.array(pre_annotation[0])
    pre_annotation = (pre_annotation - np.min(pre_annotation)) \
                     / (np.max(pre_annotation) - np.min(pre_annotation))
    pre_annotation = np.clip(pre_annotation * 255.0, 0, 255.0)
    pre_annotation = pre_annotation.astype(np.uint8).reshape((224, 224, 3))

    cv2.imshow("generated", pre_annotation)
    cv2.imshow("image", train_image[0])
    cv2.imshow("ground_truth", annotation[0])

def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG model params not found")
    data = scipy.io.loadmat(filepath)
    return data

def xavier_init(inputs, outputs, constant=1):
    # Xavier initialization
    low = -constant * np.sqrt(6.0 / (inputs + outputs))
    high = constant * np.sqrt(6.0 / (inputs + outputs))
    return tf.random_uniform((inputs, outputs), minval=low, maxval=high, dtype=tf.float32)

def get_weights_variable(inputs, outputs, name):
    w = tf.Variable(xavier_init(inputs, outputs), name=name)
    return w

def get_bias_variable(num, name):
    b = tf.Variable(tf.zeros([num], dtype=tf.float32), name=name)
    return b

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var

def weights_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name=name, initializer=initial)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] *= 2
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(conv, b)

def conv2d_basic(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.bias_add(conv, b)

def conv2d_strided(x, W, b, stride=None, padding='SAME'):
    if stride is None:
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    else:
        conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.bias_add(conv, b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def process_image(image, mean):
    return image - mean

def add_grad_summary(grad, var):
    return tf.summary.histogram(var.op.name+'gradient', grad)

def save_image(image, save_dir, name, mean=None):
    """
    save the image
    :param image:
    :param save_dir:
    :param name:
    :param mean:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name+'.png'), image)

def unprocess_image(image, mean):
    return image+mean

def to_categorial(labels):
    one_hot = np.zeros(labels.shape[0], labels.max() + 1)
    one_hot[np.array(labels.shape[0], labels)] = 1
    return one_hot

def compute_euclidean_distance(x, y, positive=True):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    d = tf.square(tf.subtract(x, y))
    d = tf.reduce_sum(d, axis=1)
    if positive:
        d1, indx = tf.nn.top_k(input=d, k=100)
    else:
        d1, indx = tf.nn.top_k(input=-d, k=100)
        d1 = -1.0 * d1

    return d1 * 2.0

def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):

    """
    Compute the contrastive loss as in
    L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m
    **Parameters**
     anchor_feature:
     positive_feature:
     negative_feature:
     margin: Triplet margin
    **Returns**
     Return the loss operation
    """

    with tf.variable_scope('triplet_loss'):
        pos_dist = compute_euclidean_distance(anchor_feature, positive_feature, positive=True)
        neg_dist = compute_euclidean_distance(anchor_feature, negative_feature, positive=False)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss, tf.reduce_mean(pos_dist), tf.reduce_mean(neg_dist)


def compute_accuracy(data_train, labels_train, data_validation, labels_validation, n_classes):

    models = []
    for i in range(n_classes):
        indexes = labels_train == i
        models.append(np.mean(data_train[indexes, :], axis=0))

    tp = 0
    for i in range(data_validation.shape[0]):
        d = data_validation[i, :]
        l = labels_validation[i]

        scores = [consine(m, d) for m in models]
        predict = np. argmax(scores)
        if predict == 1:
            tp += 1
    return (float(tp) / data_validation.shape[0]) * 100

def get_index(labels, val):
    return [i for i in range(len(labels)) if labels[i] == val]

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def whiten(x):
    mean = np.mean(x)
    std = np.std(x)
    y = np.multiply(np.subtract(x, mean), 1.0 / std)
    return y

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1-sz2+v):(sz1+sz2+v + 1), (sz1-sz2+h):(sz1+sz2+h + 1), :]
    return image

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

def random_crop(img, image_size):
    width = height = image_size
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)

    return img[y:y+height, x:x+width]


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.reduce_mean(tf.square(features - centers_batch))

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

# def plot_embedding(X, y, title=None):
#     x_min, x_max = bp.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#
#     plt.figure(figsize=(10, 10))
#     ax = plt.subplot(111)
#     for i in range(X.shape[0]):
#         plt.text(X[i, 0], X[i, 1], str(y[i]), color=cm.Set1(y[i] / 10.0), fontdict={'weight': 'bold', 'size': 9})
#
#     plt.xticks([]), plt.yticks([])
#     if title is not None:
#         plt.title(title)