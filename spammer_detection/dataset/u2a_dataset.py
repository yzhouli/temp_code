import json
import os

import numpy as np
import tensorflow as tf

from util.participant_behavior import ParticipantBehavior
from util.self_behavior import SelfBehavior


class User2AudioDataset(object):

    def __init__(self, mode, path_ref, batch_size, event_size, time_interval, event_threshold,
                 user_threshold, decay_factor, audio_sampling_num, hz):
        self.path_ref = path_ref
        self.batch_size = batch_size
        self.sb = SelfBehavior(event_size=event_size, time_interval=time_interval)
        self.pb = ParticipantBehavior(event_threshold=event_threshold, user_threshold=user_threshold,
                                      decay_factor=decay_factor, audio_sampling_num=audio_sampling_num, hz=hz,
                                      time_interval=time_interval)
        if 'train' == mode:
            self.data = json.load(open(f'{path_ref}/train.json'))
        else:
            self.data = json.load(open(f'{path_ref}/test.json'))

    def process_index(self, index):
        index = tf.cast(index, tf.int32)
        return index

    def get_all(self):
        index_li = np.asarray([int(i) for i in self.data])
        data_db = tf.data.Dataset.from_tensor_slices(index_li)
        data_db = data_db.map(self.process_index).shuffle(10000).batch(self.batch_size)
        return data_db

    def get_item(self, index_li):
        index_li = index_li.numpy()
        self_matrix, participant_matrix, topic_label = [], [], []
        for index in index_li:
            self_behaviour, participant_behaviour, label = self.iteration(index=index)
            self_matrix.append(self_behaviour)
            participant_matrix.append(participant_behaviour)
            topic_label.append(label)
        self_matrix = np.asarray(self_matrix, dtype=np.float32)
        participant_matrix = np.asarray(participant_matrix, dtype=np.float32)
        topic_label = np.asarray(topic_label, dtype=np.int32)
        return self_matrix, participant_matrix, topic_label

    def iteration(self, index):
        index = str(index)
        user_path = self.path_ref + '/' + self.data[index]['path']
        self_behaviour = self.sb.wv_self_behaviour(user_path=user_path)
        participant_behaviour = self.pb.build_participant_feature(user_path=user_path, is_audio=True, is_dwt=True)
        label = self.data[index]['label']
        return self_behaviour, participant_behaviour, label

    def len(self):
        return len(self.data.keys())
