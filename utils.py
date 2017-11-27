import tensorflow as tf
import numpy as np
import os, sys
import scipy.misc as misc
import tarfile
import zipfile
import scipy.io
import urllib.request
import cv2

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
        file_path, _ = urllib.request.urlretrieve(model_url, filepath, reporthook=_progress)
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