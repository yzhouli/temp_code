from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, models

from dataset.u2a_dataset import User2AudioDataset
from util.data_util import DataUtil


class User2AudioTrain(object):

    def __init__(self, path_ref, learning_rate, model, epochs, batch_size, event_size, time_interval,
                 event_threshold, user_threshold, decay_factor, audio_sampling_num, hz):
        self.path_ref = path_ref
        self.learning_rate = learning_rate
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.event_size = event_size
        self.time_interval = time_interval
        self.event_threshold = event_threshold
        self.user_threshold = user_threshold
        self.decay_factor = decay_factor
        self.audio_sampling_num = audio_sampling_num
        self.hz = hz
        self.train_db, self.test_db = None, None
        self.softmax = tf.keras.layers.Softmax()

    def data_init(self):
        self.train_db = User2AudioDataset(mode='train', path_ref=self.path_ref, batch_size=self.batch_size,
                                          event_size=self.event_size,
                                          time_interval=self.time_interval,
                                          event_threshold=self.event_threshold, user_threshold=self.user_threshold,
                                          decay_factor=self.decay_factor, audio_sampling_num=self.audio_sampling_num,
                                          hz=self.hz)

        self.test_db = User2AudioDataset(mode='test', path_ref=self.path_ref, batch_size=self.batch_size,
                                         event_size=self.event_size,
                                         time_interval=self.time_interval,
                                         event_threshold=self.event_threshold, user_threshold=self.user_threshold,
                                         decay_factor=self.decay_factor, audio_sampling_num=self.audio_sampling_num,
                                         hz=self.hz)

    def iteration(self, name, epoch):
        data_train = self.train_db
        if 'test' == name:
            data_train = self.test_db
        optimizer = optimizers.Adam(self.learning_rate)
        loss_num, acc = 0, 0
        pred_matrix, label_li = None, []
        pbar = tqdm(total=data_train.len() // self.batch_size + 1)
        with tf.device('/gpu:0'):
            for item in data_train.get_all():
                att_matrix, rel_matrix, topic_label = data_train.get_item(index_li=item)
                with tf.GradientTape() as tape:
                    label_matrix = tf.one_hot(topic_label, depth=2)
                    out_matrix = self.model((att_matrix, rel_matrix))
                    loss = tf.losses.categorical_crossentropy(label_matrix, out_matrix, from_logits=True)
                    loss = tf.reduce_mean(loss)
                    loss_num += float(loss) * self.batch_size
                    out_matrix = self.softmax(out_matrix)
                    if pred_matrix is None:
                        pred_matrix = out_matrix
                    else:
                        pred_matrix = tf.concat(values=[pred_matrix, out_matrix], axis=0)
                    [label_li.append(i) for i in topic_label]
                    accuracy = DataUtil.acc(label_matrix=topic_label, out_matrix=out_matrix)
                    acc += accuracy
                    pbar.desc = f'epoch: {epoch}, name: {name}, loss: {round(loss_num / len(label_li), 4)}, accuracy: {round(acc / len(label_li), 4)}'
                    pbar.update(1)
                    if 'train' == name:
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        label_matrix = tf.one_hot(label_li, depth=2)
        del tape
        hot_label_li, hot_pred_li = DataUtil.list2_list1(list2=label_matrix), DataUtil.list2_list1(list2=pred_matrix)
        auc_score = DataUtil.auc_compute(real_li=hot_label_li, pred_li=hot_pred_li)
        fpr, tpr = DataUtil.roc_compute(real_li=hot_label_li, pred_li=hot_pred_li)
        fpr, tpr = fpr[1], tpr[1]
        print(f'auc score: {round(auc_score, 4)}, tpr: {round(tpr, 4)}, fpr: {round(fpr, 4)}')
        accuracy = DataUtil.acc(label_matrix=label_li, out_matrix=pred_matrix)
        precision, recall, f1_score = DataUtil.evaluation(y_test=label_matrix, y_predict=pred_matrix)
        acc, loss = accuracy / data_train.len(), loss_num / data_train.len()
        print(f'accuracy: {round(acc, 4)}, loss: {round(loss, 4)}')
        print(
            f'non-precision: {round(precision[0], 4)}, non-recall: {round(recall[0], 4)}, non-f1_score: {round(f1_score[0], 4)}')
        print(
            f'spammer-precision: {round(precision[1], 4)}, spammer-recall: {round(recall[1], 4)}, spammer-f1_score: {round(f1_score[1], 4)}')
        return acc, loss, precision, recall, f1_score, auc_score, fpr, tpr

    def train(self):
        self.data_init()
        acc_max, auc_max = -1, -1
        for epoch in range(self.epochs):
            self.iteration(name='train', epoch=epoch)
            acc, loss, precision, recall, f1_score, auc_score, fpr, tpr = self.iteration(name='test', epoch=epoch)
            if acc > acc_max:
                acc_max = acc
                auc_max = auc_score
        print(f'max acc: {acc_max}, max aux: {auc_max}')
        return acc_max, auc_max
