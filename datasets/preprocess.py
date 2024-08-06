import os

from util.config import Config
from util.text_clear import line_clear

event_id = '2660439401'
path = f'/weibo/original/spammer/{event_id}'
target_path = f'weibo/dataset/spammer/{event_id}'
if not os.path.exists(target_path):
    os.mkdir(target_path)

self_path = f'{path}/self.txt'


def embed_content(content):
    content = line_clear(content)
    embed = Config.bert_embed(content, return_tensors='tf', padding=True, truncation=True)
    embed = Config.bert_model(embed)
    embed = [float(i) for i in list(embed[1][0].numpy())]
    embed = str(embed)[1:-2]
    return embed


embed_li = []
index = 0
with open(self_path, encoding='utf-8') as f:
    line = f.readline()
    while line:
        line = line.replace('\n', '')
        b_id, content = line.split('# #')
        index += 1
        print(index)
        embed = embed_content(content=content)
        line = f'{b_id}# #{embed}'
        embed_li.append(line)
        line = f.readline()

save_path = f'{target_path}/self_wv.txt'
with open(save_path, 'w+', encoding='utf-8') as f:
    for embed in embed_li:
        f.write(f'{embed}\n')

participant_path = f'{path}/participant.txt'
save_path = f'{target_path}/participant.txt'

with open(save_path, 'w+', encoding='utf-8') as fw:
    with open(participant_path, encoding='utf-8') as fr:
        line = fr.readline()
        while line:
            fw.write(line)
            line = fr.readline()
