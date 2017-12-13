import tensorflow as tf
import model
import squeeze_net
import utils
import data_process
import datetime
from DataShuffle import DataShuffle
slim = tf.contrib.slim

def main(argv=None):
    with tf.Graph().as_default() as graph:

        Margin = 0.25
        BATCH_SIZE = 40
        ITERATION = 2000000
        data_num = 16640
        train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
        train_positive_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
        train_negative_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
        # labels_anchor = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        # labels_positive = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        # labels_negative = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        tf.summary.image('input_image', train_anchor_data, 10)

        train_data = tf.concat([train_anchor_data, train_positive_data, train_negative_data], axis=0)

        pre_logits1 = model.NetworkModel(train_data, keep_prob=0.80)
        # pre_logits2 = model.NetworkModel(train_positive_data, keep_prob=0.80, reuse=True)
        # pre_logits3 = model.NetworkModel(train_negative_data, keep_prob=0.80, reuse=True)

        pre_logits1 = tf.nn.l2_normalize(pre_logits1, 1, 1e-10, name='embeddings')
        # pre_logits2 = tf.nn.l2_normalize(pre_logits2, 1, 1e-10, name='embeddings')
        # pre_logits3 = tf.nn.l2_normalize(pre_logits3, 1, 1e-10, name='embeddings')
        # print(logits.get_shape().as_list())

        vgg_train_anchor, vgg_train_positive, vgg_train_negative = tf.unstack(tf.reshape(pre_logits1, [-1, 3, 256]), 3, 1)

        loss, positives, negatives = utils.compute_triplet_loss(vgg_train_anchor, vgg_train_positive, vgg_train_negative, Margin)
        regularizer_losses = tf.add_n(tf.get_collection("losses"))
        total_loss = loss + regularizer_losses
        # print(total_loss)

        # num_batches_per_epoch = int(data_num / BATCH_SIZE * 3)
        # num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        # decay_steps = int(2 * num_steps_per_epoch)
        global_step = tf.Variable(tf.constant(0, dtype=tf.int64))
        lr = tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=global_step,
            decay_steps=data_num // BATCH_SIZE,
            decay_rate=0.96,
            staircase=True
            )

        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss, global_step=global_step)

        Saver = tf.train.Saver()

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('positives', positives)
        tf.summary.scalar('negatives', negatives)
        # tf.summary.scalar('pos', pos)
        # tf.summary.scalar('neg', neg)
        merged = tf.summary.merge_all()

        data, labels = data_process.input_data()
        print(data.shape)
        print(labels.shape)
        dataShufflu = DataShuffle(data, labels)

        with tf.Session() as sess:
            train_write = tf.summary.FileWriter('/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard2/', sess.graph)

            sess.run(tf.global_variables_initializer())
            # Saver.restore(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard2/')
            for step in range(ITERATION):
                batch_anchor, batch_positive, batch_negative, batch_labels_anchor, batch_labels_positive,\
                batch_labels_negative = dataShufflu.get_triplet(n_labels=30, n_triplet=BATCH_SIZE)

                feed_dict = {
                    train_anchor_data: batch_anchor,
                    train_positive_data: batch_positive,
                    train_negative_data: batch_negative
                    # labels_anchor: batch_labels_anchor,
                    # labels_positive: batch_labels_positive,
                    # labels_negative: batch_labels_negative
                }

                _, l, summary, Loss = sess.run([train_op, total_loss, merged, loss], feed_dict=feed_dict)

                print("%d the %s train reslut" % (step, datetime.datetime.now()))
                print('the triplet loss %g' % Loss)
                # print('the triplet total loss %g' % l)

                train_write.add_summary(summary, step)

                if step % 200 == 0:
                    Saver.save(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard2/')

            train_write.close()

if __name__ == '__main__':
    main()