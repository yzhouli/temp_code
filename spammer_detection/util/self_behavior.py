import numpy as np


# from transformers import BertTokenizer, TFBertModel

class SelfBehavior(object):
    # bert_embed = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def __init__(self, event_size, time_interval):
        self.event_size = event_size
        self.time_interval = time_interval

    # def content_embed(self, content):
    #     embed = SelfBehavior.bert_embed(content, return_tensors='tf', padding=True, truncation=True)
    #     embed = SelfBehavior.bert_model(embed)
    #     embed = list(embed[1][0].numpy())
    #     return embed

    def load_data(self, self_path):
        self_li = []
        with open(self_path) as f:
            line = f.readline()
            while line:
                line = line.replace('\n', '')
                att_li = line.split('# #')
                self_li.append([float(att_li[0]), [float(i) for i in att_li[-1].split(',')]])
                line = f.readline()
        self_li.sort(key=lambda x: x[0])
        if len(self_li) >= 2:
            start_time, end_time = self_li[0][0], self_li[-1][0]
            time_length = end_time - start_time
            time_size = time_length * self.time_interval
            start_time = end_time - time_size
            self_li = [i for i in self_li if i[0] >= start_time]
        return self_li

    def wv_self_behaviour(self, user_path):
        self_path = f'{user_path}/self_wv.txt'
        self_li = self.load_data(self_path)
        wv_li = []
        for index, [_, context_embed] in enumerate(self_li):
            if index >= self.event_size:
                break
            wv_li.append(context_embed)
        wv_padding = [0 for i in range(768)]
        while len(wv_li) < self.event_size:
            wv_li.append(wv_padding)
        wv_li = np.asarray(wv_li, dtype=np.float32)
        return wv_li
