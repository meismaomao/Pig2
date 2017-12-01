import tensorflow as tf
import model
import utils
import data_process
import datetime
from DataShuffle import DataShuffle

def main(argv=None):

    margin = 0.01
    BATCH_SIZE = 32
    ITERATION = 20000
    data_num = 16813
    train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
    train_positive_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='positive')
    train_negative_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name="negative")
    labels_anchor = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    labels_positive = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    labels_negative = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_probability')

    vgg_train_anchor = model.NetworkModel(train_anchor_data, keep_prob=keep_prob)
    vgg_train_positive = model.NetworkModel(train_positive_data, keep_prob=keep_prob, reuse=True)
    vgg_train_negative = model.NetworkModel(train_negative_data, keep_prob=keep_prob, reuse=True)

    loss, positives, negatives = utils.compute_triplet_loss(vgg_train_anchor,
                                                           vgg_train_positive, vgg_train_negative, margin)

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001,
        batch * BATCH_SIZE,
        data_num,
        0.95
    )

    optimizer_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
    Saver = tf.train.Saver()
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('positives', positives)
    tf.summary.scalar('negatives', negatives)
    tf.summary.scalar('lr', learning_rate)
    merged = tf.summary.merge_all()

    data, labels = data_process.input_data()
    print(data.shape)
    print(labels.shape)
    dataShufflu = DataShuffle(data, labels)

    with tf.Session() as sess:
        train_write = tf.summary.FileWriter('./logs_tensorboard/', sess.graph)
        # test_write = tf.summary.FileWriter('./logs_tensorboard/triple/test/', sess.graph)

        sess.run(tf.global_variables_initializer())

        for step in range(ITERATION):
            batch_anchor, batch_positive, batch_negative,\
            batch_labels_anchor, batch_labels_positive,\
            batch_labels_negative = dataShufflu.get_triplet(n_labels=30, n_triplet=BATCH_SIZE)

            feed_dict = {
                train_anchor_data: batch_anchor,
                train_positive_data: batch_positive,
                train_negative_data: batch_negative,
                labels_anchor: batch_labels_anchor,
                labels_positive: batch_labels_positive,
                labels_negative: batch_labels_negative,
                keep_prob: 0.8
            }
            _, l, lr, summary = sess.run([optimizer_op, loss, learning_rate, merged],
                                         feed_dict=feed_dict)

            print("%d the %s train reslut" % (step, datetime.datetime.now()))
            print('the triplet loss %g' % l)


            train_write.add_summary(summary, step)

            if step % 100 == 0:
                Saver.save(sess, './logs_tensorboard/')

        train_write.close()

if __name__ == '__main__':
    main()