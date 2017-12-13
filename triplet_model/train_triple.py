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

        margin = 0.20
        BATCH_SIZE = 30
        ITERATION = 2000000
        data_num = 16640
        train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
        train_positive_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
        train_negative_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
        labels_anchor = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        labels_positive = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        labels_negative = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
        tf.summary.image('input_image', train_anchor_data, 10)
        train_data = tf.concat([train_anchor_data, train_positive_data, train_negative_data], axis=0)

        # checkpoint_file = tf.train.latest_checkpoint('/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard/')
        # print(checkpoint_file)

        with slim.arg_scope(inception_resnet_v2_slim_version.inception_resnet_v2_arg_scope()):
            pre_logits, logits = inception_resnet_v2_slim_version.inception_resnet_v2(train_data,
                                                                                      num_classes=128, is_training=False)

        pre_logits = tf.nn.l2_normalize(pre_logits, 1, 1e-12, name='embeddings')
        # print(logits.get_shape().as_list())

        # exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=)

        vgg_train_anchor, vgg_train_positive, vgg_train_negative = tf.unstack(tf.reshape(pre_logits, [-1, 3, 128]), 3, 1)

        loss, positives, negatives = utils.compute_triplet_loss(vgg_train_anchor,
                                                               vgg_train_positive, vgg_train_negative, margin)
        slim.losses.add_loss(loss)
        total_loss = slim.losses.get_total_loss()

        num_batches_per_epoch = int(data_num / BATCH_SIZE * 3)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(2 * num_steps_per_epoch)
        global_step = tf.Variable(0)
        lr = tf.train.exponential_decay(
            0.01,
            global_step * BATCH_SIZE,
            data_num,
            0.95
            )

        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss)

        Saver = tf.train.Saver(variables_to_restore)

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', loss)
        # tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('positives', positives)
        tf.summary.scalar('negatives', negatives)
        merged = tf.summary.merge_all()

        data, labels = data_process.input_data()
        print(data.shape)
        print(labels.shape)
        dataShufflu = DataShuffle(data, labels)

        with tf.Session() as sess:
            train_write = tf.summary.FileWriter('/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard/logs_tensorboard', sess.graph)

            sess.run(tf.global_variables_initializer())
            Saver.restore(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard/logs_tensorboard/model-ckpt-1')
            for step in range(ITERATION):
                batch_anchor, batch_positive, batch_negative, batch_labels_anchor, batch_labels_positive,\
                batch_labels_negative = dataShufflu.get_triplet(n_labels=30, n_triplet=BATCH_SIZE)

                feed_dict = {
                    train_anchor_data: batch_anchor,
                    train_positive_data: batch_positive,
                    train_negative_data: batch_negative,
                    labels_anchor: batch_labels_anchor,
                    labels_positive: batch_labels_positive,
                    labels_negative: batch_labels_negative
                }

                l,summary, Loss = sess.run([total_loss, merged, loss], feed_dict=feed_dict)

                print("%d the %s train reslut" % (step, datetime.datetime.now()))
                print('the triplet loss %g' % Loss)
                print('the triplet total loss %g' % l)

                train_write.add_summary(summary, step)

                # if step % 100 == 0:
                #     Saver.save(sess, '/home/lenovo/yql/pig_data/triplet_model/logs_tensorboard/logs_tensorboard/'
                #                + 'model.ckpt', global_step=step+1)

            train_write.close()

if __name__ == '__main__':
    main()