import tensorflow as tf
import numpy as np
import utils
import data_process
from flip_gradient import flip_gradient
import datetime


MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
CLASS_COUNTS = 100
BATCH_SIZE = 128
MAX_ITRATIONS = int(1e10 + 1)
IMAGE_SIZE = 32
IMAGE_CHANNLES = 3
RAW_NUM = 60000
EPOCH = 30
EPOCH1 = 10
TRAIN_NUM = 50000
EPS = 1e-10


def subnet1(image, reuse=False):
    with tf.variable_scope("subnet1", reuse=reuse):
        def _vgg_net(weights, image):
            print('setting up vgg model initialized params --> extractor1')
            layers = (
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                'relu5_3', 'conv5_4', 'relu5_4'
            )
            net = {}
            current = image
            for i, name in enumerate(layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = weights[i][0][0][0][0]
                    # kernels are [width, height, in_channles, out_channles]
                    # tensorflow are [height, width, in channles, out_channles]
                    kernels = utils.get_variable(
                        np.transpose(kernels, (1, 0, 2, 3)), name=name + '_w')
                    bias = utils.get_variable(bias.reshape(-1), name=name + '_b')
                    current = utils.conv2d_basic(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current, name=name)
                elif kind == 'pool':
                    current = utils.avg_pool_2x2(current)
                net[name] = current
            return net

        model_data = utils.get_model_data("data", MODEL_URL)
        weights = np.squeeze(model_data['layers'])

        with tf.variable_scope('inference'):
            image = tf.reshape(image, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNLES])
            image_net = _vgg_net(weights, image)
            conv_final_layer = image_net['relu5_4']

            pool5 = utils.max_pool_2x2(conv_final_layer)

            W6 = utils.weights_variable([4, 4, 512, 4096], name="W6")
            b6 = utils.bias_variable([4096], name='b6')
            conv6 = utils.conv2d_strided(pool5, W6, b6, stride=1)
            relu6 = tf.nn.relu(conv6, name='relu6')

            relu6 = utils.max_pool_2x2(relu6)

            disc_out = tf.reshape(relu6, [-1, 4096])

            return disc_out

def Domain1(features, grad_scale, keep_prob):
    with tf.name_scope('Domain1'):
        features = flip_gradient(features, grad_scale)

        W1 = utils.get_weights_variable(4096, 1024, name='W1')
        b1 = utils.get_bias_variable(1024, name='b1')
        matmul1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(features, W1), b1))

        matmul1 = tf.nn.dropout(matmul1, keep_prob)

        W2 = utils.get_weights_variable(1024, 2, name='W2')
        b2 = utils.get_bias_variable(2, name='b2')
        matmul2 = tf.nn.bias_add(tf.matmul(matmul1, W2), b2)
        return matmul2

def classifier1(feature, keep_prob):
    with tf.name_scope('classifier1'):
        W1 = utils.get_weights_variable(4096, 1024, name='W1')
        b1 = utils.get_bias_variable(1024, name='b1')
        matmul1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(feature, W1), b1))

        matmul1 = tf.nn.dropout(matmul1, keep_prob)

        W2 = utils.get_weights_variable(1024, CLASS_COUNTS, name='W2')
        b2 = utils.get_bias_variable(CLASS_COUNTS, name='b2')
        matmul2 = tf.nn.bias_add(tf.matmul(matmul1, W2), b2)
        return matmul2

def accuracy_fun(labels, logits):
    logits = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

def main(argv=None):

    train_samples = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNLES], name='train_sample')
    raw_samples = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNLES], name='raw_sample')
    train_labels = tf.placeholder(tf.float32, [None, CLASS_COUNTS], name='train_label')
    raw_labels = tf.placeholder(tf.float32, [None, 2], name='raw_label')
    grad_scale = tf.placeholder(tf.float32, name='grad_scale')
    keep_prob = tf.placeholder(tf.float32, [], name='keep_probability')

    train_feature1 = subnet1(train_samples)

    clas1_train_feature1_logits = classifier1(train_feature1, keep_prob)

    raw_feature1 = subnet1(raw_samples, reuse=True)

    dom1_raw_feature1_logits = Domain1(raw_feature1, grad_scale, keep_prob)

    ##WGANs loss function about discriminator model
    # d_var = [var for var in tf.trainable_variables() if var.name.startswith('Discriminator1')]
    # disc_extr1_loss = -1.0 * tf.reduce_mean(disc1_train_feature1_logits)
    # disc_extr2_loss = -1.0 * tf.reduce_mean(disc1_train_feature2_logits)
    # disc_loss = -1.0 * (tf.reduce_mean(disc1_train_feature1_logits) - tf.reduce_mean(disc1_train_feature2_logits))
    # d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_var]

    clas1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=clas1_train_feature1_logits))

    dom1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=raw_labels, logits=dom1_raw_feature1_logits))

    clas1_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_labels, 1),
                                                tf.argmax(clas1_train_feature1_logits, 1)), tf.float32), name='clas1_acc')
    classifier1_loss = clas1_loss + dom1_loss


    classifier1_op = tf.train.AdamOptimizer(0.0002).minimize(classifier1_loss)


    train_data, train_label, t_test, t_label, raw_data, raw_label = data_process.input_data()
    print(np.shape(t_test))
    print(np.shape(t_label))

    Saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./checkpoint/', sess.graph)
        sess.run(tf.global_variables_initializer())
        # Saver.restore(sess, './checkpoint/')
        print('Initialized!')
        for step in range(int(RAW_NUM * EPOCH) // BATCH_SIZE):
            print('begin path classifier train')
            train_offset = (step * BATCH_SIZE) % (TRAIN_NUM - BATCH_SIZE)
            batch_data = train_data[train_offset:(train_offset + BATCH_SIZE), :]
            batch_label = train_label[train_offset:(train_offset + BATCH_SIZE), :]

            raw_offset = (step * BATCH_SIZE) % (RAW_NUM - BATCH_SIZE)
            raw_batch_data = raw_data[raw_offset:(raw_offset + BATCH_SIZE), :]
            raw_batch_label = raw_label[raw_offset:(raw_offset + BATCH_SIZE), :]

            feed_dict = {train_samples: batch_data, train_labels: batch_label,
                         raw_samples: raw_batch_data, raw_labels: raw_batch_label, grad_scale: -1.0, keep_prob: 0.85}

            # sess.run(d_clip)
            _, Clas1_loss = sess.run([classifier1_op, classifier1_loss], feed_dict=feed_dict)

            Clas1_acc, Clas2_acc = sess.run([clas1_acc], feed_dict=feed_dict)

            print("%d the %s train reslut" % (step, datetime.datetime.now()))
            print('the classifier one loss %g' % Clas1_loss)

            print('the classifier1 accuracy is %g, the classifier2 accuracy is %g' % (Clas1_acc, Clas2_acc))

            if step % 500 == 0:
                Saver.save(sess, './checkpoint/')


if __name__ == '__main__':
    main()