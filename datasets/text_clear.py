import os
import re


def substr(text, start=0, end=0):
    if end == 0:
        end = len(text) - 1
    temp = ''
    for i in range(start, end + 1):
        temp += text[i]
    return temp


def findstr(text: str, target, start=0):
    if start == 0:
        return text.index(target)
    elif start > len(text):
        return -1
    else:
        temp = ''
        for i in range(start, len(text)):
            temp += text[i]
        if temp.__contains__(target):
            return temp.index(target) + start
        return -1


def remove_reply(text: str):
    if text.__contains__('回复'):
        text = text.replace('回复', '')
    if text.__contains__('@'):
        start_index = findstr(text, '@')
        end_index = findstr(text, ':', start=start_index + 1)
        if end_index > 0:
            temp = substr(text, start_index, end_index)
            text = text.replace(temp, '')
            text = remove_reply(text)
    if text.__contains__('@'):
        start_index = findstr(text, '@')
        end_index = findstr(text, '：', start=start_index + 1)
        if end_index > 0:
            temp = substr(text, start_index, end_index)
            text = text.replace(temp, '')
            text = remove_reply(text)
    if text.__contains__('@'):
        start_index = findstr(text, '@')
        end_index = findstr(text, ' ', start=start_index + 1)
        if end_index > 0:
            temp = substr(text, start_index, end_index)
            text = text.replace(temp, '')
            text = remove_reply(text)
    while '<' in text:
        start_index = findstr(text, '<')
        end_index = findstr(text, '>', start=start_index + 1)
        if end_index <= start_index:
            break
        temp = substr(text, start_index, end_index)
        text = text.replace(temp, '')
        text = remove_reply(text)
    return text


def remove(text: str, type='//'):
    if text.__contains__(type):
        end = text.index(type)
        temp = ''
        for i in range(end):
            temp += text[i]
        return temp
    return text


def remove_retweet(text, type='//'):
    if text == '':
        return text
    result = remove(text, type=type)
    if len(result) <= 0:
        temp = ''
        for i in range(len(type), len(text)):
            temp += text[i]
        text = remove_retweet(temp)
        return text
    return result


def text_clear(source_path):
    file_line = []
    with open(source_path) as f:
        line = f.readline()
        while line:
            line = line.split('\n')[0].split('@@##$$%%&&')
            file_line.append(line)
            line = f.readline()

    temp_line = []
    for i in range(len(file_line)):
        text = file_line[i][1]
        # 删除文本中的URL
        text = re.sub(r'http://\S+|https://\S+', '', text, flags=re.MULTILINE)
        # 删除文本中的@信息
        text = remove_reply(text)
        # 删除文本中的转发信息
        text = remove_retweet(text)
        text = text.replace(' ', '')
        if text != '':
            print(i, text)
            temp_line.append((file_line[i][0], text))

    with open(source_path, 'w+') as f:
        for i in temp_line:
            f.write(i[0] + '@@##$$%%&&' + i[1] + '\n')
    if len(temp_line) < 70:
        print(len(temp_line))


def copy_to_word(path, word_path):
    file_line = []
    with open(path) as f:
        line = f.readline()
        while line:
            line = line.split('\n')[0].split('@@##$$%%&&')
            file_line.append(line)
            line = f.readline()
    with open(word_path, 'a+') as f:
        for i in file_line:
            text = i[1]
            f.write(text + '\n')


def line_clear(text):
    # 删除文本中的URL
    text = re.sub(r'http://\S+|https://\S+', '', text, flags=re.MULTILINE)
    # 删除文本中的@信息
    text = remove_reply(text)
    # 删除文本中的转发信息
    text = remove_retweet(text)
    text = text.replace(' ', '')
    return text
