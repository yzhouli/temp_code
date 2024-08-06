import math

import jieba
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


class DataUtil(object):

    @staticmethod
    def compute_entropy(emo_li):
        result = 0
        for i in emo_li:
            result -= i * math.log2(i)
        result -= 0.000001 * math.log2(0.000001)
        return result

    @staticmethod
    def max_index(pred_li):
        index = 0
        temp = -1000
        for i, num in enumerate(pred_li):
            if num > temp:
                temp = num
                index = i
        return index

    @staticmethod
    def acc(label_matrix, out_matrix):
        out_matrix = out_matrix.numpy()
        true_total = 0
        for i in range(len(label_matrix)):
            if label_matrix[i] == -1:
                continue
            pred = tf.nn.softmax(out_matrix[i])
            pred_index = DataUtil.max_index(pred_li=pred)
            if pred_index == label_matrix[i]:
                true_total += 1
        return true_total

    @staticmethod
    def normal(predict_li, depth=3):
        result_li = []
        for att_li in predict_li.numpy():
            index = DataUtil.max_index(att_li)
            result_li.append(index)
        result_li = np.asarray(result_li, dtype=np.int32)
        result_li = tf.cast(result_li, dtype=tf.int32)
        result_li = tf.one_hot(result_li, depth=depth)
        return result_li

    @staticmethod
    def evaluation(y_test, y_predict):
        y_predict = DataUtil.normal(predict_li=y_predict, depth=2)
        metrics = classification_report(y_test, y_predict, output_dict=True)
        precision = metrics['0']['precision'], metrics['1']['precision']
        recall = metrics['0']['recall'], metrics['1']['recall']
        f1_score = metrics['0']['f1-score'], metrics['1']['f1-score']
        return precision, recall, f1_score

    @staticmethod
    def auc_compute(pred_li, real_li):
        auc = roc_auc_score(y_true=real_li, y_score=pred_li)
        return auc

    @staticmethod
    def roc_compute(pred_li, real_li):
        fpr, tpr, thresholds_keras = roc_curve(y_true=real_li, y_score=pred_li)
        return fpr, tpr

    @staticmethod
    def list2_list1(list2: list[list]):
        list1 = []
        for list_item in list2:
            for num_tensor in list_item:
                list1.append(float(num_tensor))
        return list1
