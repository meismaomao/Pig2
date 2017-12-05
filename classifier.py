import tensorflow as tf
import numpy as np
import model
import data_process
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main(data, iter_num, labels, train=True):
    BATCH_SIZE = 2
    ITERATION = iter_num
    embedding_size = 1024
    CLASS_NUM = 30

    train_anchor_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3], name='anchor')

    labels_anchor = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_probability')

    pre_logits = model.NetworkModel(train_anchor_data, keep_prob=keep_prob)
    logits = tf.nn.l2_normalize(pre_logits, 1, 1e-12, name='embeddings')
    # print(logits.get_shape().as_list())
    data_train, train_labels = data, labels
    # data_train = data

    print(data.shape)
    print(labels.shape)
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        # test_write = tf.summary.FileWriter('./logs_tensorboard/triple/test/', sess.graph)

        sess.run(tf.global_variables_initializer())
        Saver.restore(sess, './logs_tensorboard2/')

        emb_array = np.zeros((ITERATION // BATCH_SIZE * BATCH_SIZE, embedding_size))
        print(np.shape(emb_array))

        for step in range(ITERATION//BATCH_SIZE):
            batch_anchor, batch_labels = data_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE], \
                                          train_labels[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            # batch_anchor = data_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

            feed_dict = {
                train_anchor_data: batch_anchor,
                labels_anchor: batch_labels,
                keep_prob: 1.0
            }

            Logits = sess.run(logits, feed_dict=feed_dict)
            emb_array[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, :] = Logits
            print('the triplet loss %g' % step)
    np.savetxt('valid_feature.txt', emb_array)
    if train:
        print('Training classifier')
        class_name = ['%2d' % (i) for i in range(CLASS_NUM)]
        print(class_name)
        # SVMmodel = SVC(kernel='rbf', probability=True)
        knn = KNeighborsClassifier(n_neighbors=30)
        print("fit !!!")
        # SVMmodel.fit(emb_array, labels[:ITERATION // BATCH_SIZE * BATCH_SIZE])
        knn.fit(emb_array, labels[:ITERATION // BATCH_SIZE * BATCH_SIZE])

        with open('train_model_pickle30', 'wb') as outfile:
            pickle.dump((knn, class_name), outfile)
            # pickle.dump((SVMmodel, class_name), outfile)
        print('Saved classifier model to file train_model_pickle')
    else:
        print('Testing classifier30')
        with open('train_model_pickle30', 'rb') as infile:
            (knn, class_name) = pickle.load(infile)
        print('Loaded classifier model from file train_model_pickle')

        predictions = knn.predict_proba(emb_array)
        np.savetxt('test_k_30_image_result.csv', predictions, fmt='%g')
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d %s %0.3f' % (i, class_name[best_class_indices[i]], best_class_probabilities[i]))
        accuracy = np.mean(np.equal(best_class_indices, labels))
        print("Accuracy %.3f" % accuracy)

if __name__ == "__main__":

    # data, label = data_process.input_data()
    # main(data=data, iter_num=16813, labels=label, train=True)
    data, label = data_process.input_valid_data()
    main(data=data, iter_num=866, labels=label, train=False)
    # data = data_process.input_test_data()
    # main(data=data, iter_num=3000, labels=None, train=False)