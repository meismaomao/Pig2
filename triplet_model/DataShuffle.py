import numpy as np
import data_process
import utils

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
            data_anchor = input_data[indexes[0], :, :, :]
            data_positive = input_data[indexes[1], :, :, :]

            indexes = utils.get_index(input_labels, index[1])
            # print(indexes)
            np.random.shuffle(indexes)
            data_negative = input_data[indexes[0], :, :, :]

            return data_anchor, data_positive, data_negative, \
                   label_positive, label_positive, label_negative

        target_data = self.train_data
        target_labels = self.train_labels
        # print(target_labels)

        c = target_data.shape[3]
        w = target_data.shape[1]
        h = target_data.shape[2]

        data_a = np.zeros(shape=(n_triplet, w, h, c), dtype=np.float32)
        data_p = np.zeros(shape=(n_triplet, w, h, c), dtype=np.float32)
        data_n = np.zeros(shape=(n_triplet, w, h, c), dtype=np.float32)
        labels_a = np.zeros(shape=n_triplet, dtype=np.float32)
        labels_p = np.zeros(shape=n_triplet, dtype=np.float32)
        labels_n = np.zeros(shape=n_triplet, dtype=np.float32)

        for i in range(n_triplet):
            data_a[i, :, :, :], data_p[i, :, :, :], data_n[i, :, :, :], \
            labels_a[i], labels_p[i], labels_n[i] = _get_one_triplet(target_data, target_labels)

        return data_a, data_p, data_n, labels_a, labels_p, labels_n


# if __name__ == '__main__':
#     BATCH_SIZE = 32
#     data, labels = data_process.input_data()
#     dataShuffle = DataShuffle(data, labels)
#     batch_anchor, batch_positive, batch_negative, \
#     batch_labels_anchor, batch_labels_positive, \
#     batch_labels_negative = dataShuffle.get_triplet(n_labels=30, n_triplet=BATCH_SIZE)