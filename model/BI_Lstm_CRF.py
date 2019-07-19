# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 17:13
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : BI_Lstm_CRF.py
# @Software: PyCharm

import keras
from keras.layers import Embedding,Bidirectional,GRU,AlphaDropout,Masking,TimeDistributed,Dropout
from keras.models import Sequential
from model.crf_layer import CRF
class BiLSTM_CRF():
    def __init__(self,n_input,n_vocab,n_embed,embeding_mat,keep_prob,
                 n_lstm,keep_prob_lstm,n_entity,optimizer,batch_size,epochs):
        self.n_input = n_input
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.embeding_mat = embeding_mat
        self.keep_prob = keep_prob
        self.n_lstm = n_lstm
        self.keep_prob_lstm = keep_prob_lstm
        self.n_entity = n_entity
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

        self.build()

    def build(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.n_vocab,
                                 output_dim=self.n_embed,
                                 input_length=self.n_input,
                                 weights=[self.embeding_mat],
                                 mask_zero=True,
                                 trainable=True))
        self.model.add(Dropout(self.keep_prob))
        self.model.add(Bidirectional(GRU(self.n_lstm,
                                         return_sequences=True,
                                         dropout=self.keep_prob_lstm,
                                         recurrent_dropout=self.keep_prob_lstm)))
        crf = CRF(units=self.n_entity,
                  learn_mode='join',
                  test_mode='viterbi',
                  sparse_target=False)
        self.model.add(crf)
        self.model.compile(optimizer=self.optimizer,
                           loss=crf.loss_function,
                           metrics=[crf.accuracy])

    def train(self,x_train,y_train,x_dev,y_dev,cb):
        self.model.fit(x_train,y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=[x_dev,y_dev],
                       callbacks=cb)