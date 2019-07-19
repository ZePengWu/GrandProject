# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 12:24
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : trainEmb.py
# @Software: PyCharm

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
def trainW2v():
    sentences = LineSentence('../datagrand/newCropus.txt')
    model = Word2Vec(sentences, size=300, window=5, min_count=1, sg=1, workers=4)  # sg=0 使用cbow训练, sg=1对低频词较为敏感
    model.save('../datagrand/w2v/grand.w2v.300d.txt')

def build_word_dict(model_path):
    """  获取word2vec模型的所有词向量:
    param model_path: 已经训练好的word2vec模型保存路径
    return:返回词向量词典
    """
    model = Word2Vec.load(model_path)
    vocab = model.wv.vocab
    word_vector = {}
    for word in vocab:
        # print (word)
        word_vector[word] = np.asarray(model[word],dtype='float32')
    return word_vector

if __name__ == '__main__':
    word_vector = build_word_dict('../datagrand/w2v/grand.w2v.300d.txt')
    print ('---')
    print (len(word_vector))
    print (word_vector['20000'])
    # trainW2v()
