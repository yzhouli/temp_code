from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.applications.resnet50 import ResNet50


class Config(object):
    bert_embed = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
