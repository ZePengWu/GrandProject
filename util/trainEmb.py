# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 12:24
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : trainEmb.py
# @Software: PyCharm

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
sentences = LineSentence('../datagrand/newCropus.txt')
model = Word2Vec(sentences, size=300, window=5, min_count=2, sg=1, workers=4)  # sg=0 使用cbow训练, sg=1对低频词较为敏感
model.save('../datagrand/w2v/grand.w2v.300d.txt')

