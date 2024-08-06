import pywt
import librosa
import numpy as np


class ParticipantBehavior(object):

    def __init__(self, event_threshold, user_threshold, decay_factor, audio_sampling_num, hz, time_interval):
        self.event_threshold = event_threshold
        self.user_threshold = user_threshold
        self.decay_factor = decay_factor
        self.audio_sampling_num = audio_sampling_num
        self.hz = hz
        self.sampling_interval = 1 / hz
        self.time_interval = time_interval

    def load_data(self, participant_path, is_audio=True):
        participant_li = []
        with open(participant_path) as f:
            line = f.readline()
            while line:
                line = line.replace('\n', '')
                att_li = line.split('# #')
                att_li = [float(i) for i in att_li]
                user_level = att_li[2] / self.user_threshold
                event_level = att_li[1] / self.event_threshold
                amplitude = user_level + self.decay_factor * event_level
                participant_li.append([att_li[0], amplitude, att_li[-1]])
                line = f.readline()
        participant_li.sort(key=lambda x: x[0])
        if len(participant_li) >= 2:
            start_time, end_time = participant_li[0][0], participant_li[-1][0]
            time_length = end_time - start_time
            time_size = time_length * self.time_interval
            start_time = end_time - time_size
            participant_li = [i for i in participant_li if i[0] >= start_time]
        if not is_audio:
            return [[i[-2], i[-1]] for i in participant_li]
        return participant_li

    def build_audio(self, participant_li):
        participant_size = max(len(participant_li), 1)
        point_num = self.audio_sampling_num // participant_size
        audio_li = []
        index = 0
        for [_, amplitude, frequency] in participant_li:
            amplitude = 1
            for i in range(point_num):
                index += self.sampling_interval
                value = amplitude * np.cos(2 * np.pi * frequency * index)
                audio_li.append(value)
        while len(audio_li) < self.audio_sampling_num:
            audio_li.append(0.0)
        return audio_li

    def get_audio_feature(self, audio_li):
        image = librosa.feature.mfcc(y=audio_li, sr=self.hz)
        return image

    def build_participant_feature(self, user_path, is_audio=False, is_dwt=False):
        participant_path = f'{user_path}/participant.txt'
        participant_li = self.load_data(participant_path=participant_path, is_audio=is_audio)
        if not is_audio:
            temp_num = 1500
            participant_li = [participant_li[i] for i in range(len(participant_li)) if i < temp_num]
            while len(participant_li) < temp_num:
                participant_li.append([0.0, 0.0])
            return participant_li
        audio_li = self.build_audio(participant_li=participant_li)
        audio_li = np.asarray(audio_li, dtype=np.float32)
        if is_dwt:
            feature, _ = pywt.cwt(audio_li, np.arange(1, 31), 'gaus1')
        else:
            feature = self.get_audio_feature(audio_li=audio_li)
        feature = np.asarray(feature, dtype=np.float32)
        return feature


