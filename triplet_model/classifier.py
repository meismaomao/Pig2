import tensorflow as tf
import numpy as np
import model
import data_process
import pickle
from sklearn.svm import SVC

def main(train=True):
    BATCH_SIZE = 1
    ITERATION = 866
    embedding_size = 1024
    CLASS_NUM = 30
    train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')
    labels_anchor = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_probability')


    pre_logits = model.NetworkModel(train_anchor_data, keep_prob=keep_prob)
    logits = tf.nn.l2_normalize(pre_logits, 1, 1e-12, name='embeddings')
    # print(logits.get_shape().as_list())
    data, labels = data_process.input_data()

    print(data.shape)
    print(labels.shape)
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        # test_write = tf.summary.FileWriter('./logs_tensorboard/triple/test/', sess.graph)

        sess.run(tf.global_variables_initializer())
        Saver.restore(sess, './logs_tensorboard/')

        emb_array = np.zeros((ITERATION, embedding_size))

        for step in range(ITERATION):
            batch_anchor, batch_labels_anchor = data[step], labels[step, :]
            feed_dict = {
                train_anchor_data: batch_anchor,
                labels_anchor: batch_labels_anchor,
                keep_prob: 1.0
            }

            logits = sess.run(logits, feed_dict=feed_dict)
            emb_array[step, :] = logits
            print('the triplet loss %g' % step)

    if train:
        print('Training classifier')
        SVMmodel = SVC(kernel='linear', probability=True)
        SVMmodel.fit(emb_array, labels)
        class_name = [ '%2d' % (i) for i in range(CLASS_NUM)]
        with open('train_model_pickle', 'wb') as outfile:
            pickle.dump((SVMmodel, class_name), outfile)
        print('Saved classifier model to file train_model_pickle')
    else:
        print('Testing classifier')
        with open('train_model_pickle', 'rb') as infile:
            (SVMmodel, class_name) = pickle.load(infile)
        print('Loaded classifier model from file train_model_pickle')

        predictions = SVMmodel.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d %s %0.3f' % (i, class_name[best_class_indices[i]], best_class_probabilities[i]))
        accuracy = np.mean(np.equal(best_class_indices, labels))
        print("Accuracy %.3f" % accuracy)

if __name__ == "__main__":
    main()
    main(False)
