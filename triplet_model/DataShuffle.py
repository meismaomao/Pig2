import numpy as np
import utils
import data_process
import cv2

class DataShuffle(object):
    def __init__(self, data, labels):
        total_samples = data.shape[0]
        indexes = np.array(range(total_samples))
        np.random.shuffle(indexes)
        self.train_data = data
        self.train_labels = labels

    def get_triplet(self, n_labels, n_triplet=1):
        def _get_one_triplet(input_data, input_labels):
            input_labels = np.array(input_labels)
            index = np.random.choice(n_labels, 2, replace=False)
            label_positive = index[0]
            label_negative = index[1]

            indexes = utils.get_index(input_labels, index[0])
            np.random.shuffle(indexes)
            # print(indexes[0])
            data_anchor = input_data[indexes[0]]
            data_anchor = _read_image(data_anchor)

            data_positive = input_data[indexes[1]]
            data_positive = _read_image(data_positive)

            indexes1 = utils.get_index(input_labels, index[1])
            # print(indexes)
            np.random.shuffle(indexes1)
            data_negative = input_data[indexes1[0]]
            data_negative = _read_image(data_negative)
            # print(np.shape(data_negative))


            return data_anchor, data_positive, data_negative, label_positive, label_positive, label_negative
        def read_batch_input(input_data, input_labels):

            input_labels = np.array(input_labels)
            index = np.random.choice(n_labels, 2, replace=False)
            label_positive = index[0]

            indexes = utils.get_index(input_labels, index[0])
            np.random.shuffle(indexes)
            data_anchor = input_data[indexes[0]]
            data_anchor = _read_image(data_anchor)

            return data_anchor, label_positive

        def _read_image(im):
            im = cv2.imread(im)
            im = utils.prewhiten(im)
            im = utils.flip(im, random_flip=True)
            im = utils.random_crop(im, image_size=299)
            im = cv2.resize(im, (128, 128))
            im = utils.random_rotate_image(im)
            return im

        target_data = self.train_data
        target_labels = self.train_labels
        data_a = np.zeros(shape=(n_triplet, 128, 128, 3), dtype=np.float32)
        # data_p = np.zeros(shape=(n_triplet, 128, 128, 3), dtype=np.float32)
        # data_n = np.zeros(shape=(n_triplet, 128, 128, 3), dtype=np.float32)
        labels_a = np.zeros(shape=n_triplet, dtype=np.float32)
        # labels_p = np.zeros(shape=n_triplet, dtype=np.float32)
        # labels_n = np.zeros(shape=n_triplet, dtype=np.float32)

        for i in range(n_triplet):
            data_a[i, :, :, :], labels_a[i] = read_batch_input(target_data, target_labels)

        return data_a, labels_a

# if __name__ == '__main__':
#     BATCH_SIZE = 30
#     data, labels = data_process.input_data()
#     dataShuffle = DataShuffle(data, labels)
#     batch_anchor, batch_positive, batch_negative, \
#     batch_labels_anchor, batch_labels_positive, \
#     batch_labels_negative = dataShuffle.get_triplet(n_labels=30, n_triplet=BATCH_SIZE)