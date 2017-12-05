import tensorflow as tf
import model
import inception_resnet_v2_slim_version
import utils
import data_process
import datetime
from DataShuffle import DataShuffle
slim = tf.contrib.slim

def main(argv=None):
    with tf.Graph().as_default() as graph:

        margin = 0.25
        BATCH_SIZE = 30
        ITERATION = 200000
        data_num = 16813
        train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 299, 299, 3], name='anchor')
        train_positive_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 299, 299, 3], name='anchor')
        train_negative_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 299, 299, 3], name='anchor')
        labels_anchor = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        labels_positive = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        labels_negative = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        # keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_probability')
        train_data = tf.concat([train_anchor_data, train_positive_data, train_negative_data], axis=0)

        # vgg_train_anchor = model.NetworkModel(train_anchor_data, keep_prob=keep_prob)
        # vgg_train_positive = model.NetworkModel(train_positive_data, keep_prob=keep_prob, reuse=True)
        # vgg_train_negative = model.NetworkModel(train_negative_data, keep_prob=keep_prob, reuse=True)
        with slim.arg_scope(inception_resnet_v2_slim_version.inception_resnet_v2_arg_scope()):
            pre_logits, logits = inception_resnet_v2_slim_version.inception_resnet_v2(train_data,
                                                                                      num_classes=512, is_training=True)


        pre_logits = tf.nn.l2_normalize(pre_logits, 1, 1e-10, name='embeddings')
        # print(logits.get_shape().as_list())

        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        vgg_train_anchor, vgg_train_positive, vgg_train_negative = tf.unstack(tf.reshape(pre_logits, [-1, 3, 512]), 3, 1)
        # print(vgg_train_anchor.get_shape().as_list())
        # print(vgg_train_positive.get_shape().as_list())
        # print(vgg_train_negative.get_shape().as_list())

        loss, positives, negatives = utils.compute_triplet_loss(vgg_train_anchor,
                                                               vgg_train_positive, vgg_train_negative, margin)

        total_loss = tf.losses.get_total_loss() + loss

        num_batches_per_epoch = int(data_num / BATCH_SIZE * 3)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(2 * num_steps_per_epoch)
        global_step = tf.Variable(0)
        lr = tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=0.7,
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # optimizer_op = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        Saver = tf.train.Saver(variables_to_restore)

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('positives', positives)
        tf.summary.scalar('negatives', negatives)
        merged = tf.summary.merge_all()

        data, labels = data_process.input_data()
        print(data.shape)
        print(labels.shape)
        dataShufflu = DataShuffle(data, labels)

        with tf.Session() as sess:
            train_write = tf.summary.FileWriter('./logs_tensorboard_3/', sess.graph)
            # test_write = tf.summary.FileWriter('./logs_tensorboard/triple/test/', sess.graph)

            sess.run(tf.global_variables_initializer())
            Saver.restore(sess, 'inception_resnet_v2_2016_08_30.ckpt')
            for step in range(ITERATION):
                batch_anchor, batch_positive, batch_negative, batch_labels_anchor, batch_labels_positive,\
                batch_labels_negative = dataShufflu.get_triplet(n_labels=30, n_triplet=BATCH_SIZE)
                # print(batch_anchor, batch_positive, batch_negative, batch_labels_anchor, batch_labels_positive, batch_labels_negative)
                feed_dict = {
                    train_anchor_data: batch_anchor,
                    train_positive_data: batch_positive,
                    train_negative_data: batch_negative,
                    labels_anchor: batch_labels_anchor,
                    labels_positive: batch_labels_positive,
                    labels_negative: batch_labels_negative
                }

                _, l, ls, summary, Loss = sess.run([train_op, total_loss, lr, merged, loss], feed_dict=feed_dict)

                print("%d the %s train reslut" % (step, datetime.datetime.now()))
                print('the triplet loss %g' % Loss)
                print('the triplet total loss %g' % l)


                train_write.add_summary(summary, step)

                if step % 100 == 0:
                    Saver.save(sess, './logs_tensorboard_3/')

            train_write.close()

if __name__ == '__main__':
    main()