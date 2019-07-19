# -*- coding: utf-8 -*-
# @Time    : 2019/7/19 15:37
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : main.py.py
# @Software: PyCharm

import numpy as np
from model.BI_Lstm_CRF import BiLSTM_CRF
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,\
                            TensorBoard
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

char_embedding_mat = np.load('datagrand/embedding_matrix.npy')

X_train = np.load('datagrand/train_x.npy')
X_dev = np.load('datagrand/dev_x.npy')
y_train = np.load('datagrand/train_y.npy')
y_dev = np.load('datagrand/dev_y.npy')

# ner_model = BiLSTM_CRF(n_input=200, n_vocab=char_embedding_mat.shape[0],
#                        n_embed=100, embedding_mat=char_embedding_mat,
#                        keep_prob=0.5, n_lstm=100, keep_prob_lstm=0.8,
#                        n_entity=7, optimizer='adam', batch_size=64, epochs=500)
ner_model = BiLSTM_CRF(n_input=200, n_vocab=char_embedding_mat.shape[0],
                       n_embed=300, embeding_mat=char_embedding_mat,
                       keep_prob=0.5, n_lstm=500, keep_prob_lstm=0.6,
                       n_entity=7, optimizer='adam', batch_size=32, epochs=500)

cp_folder, cp_file = 'checkpoints', 'bilstm_crf_weights_best.hdf5'
log_filepath = '/embeding_mat/bilstm_crf_summaries'

cb = [ModelCheckpoint(os.path.join(cp_folder, cp_file), monitor='val_loss',
                      verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
      EarlyStopping(min_delta=1e-8, patience=10, mode='min'),
      ReduceLROnPlateau(factor=0.2, patience=6, verbose=0, mode='min',
                        epsilon=1e-6, cooldown=4, min_lr=1e-10),
      TensorBoard(log_dir=log_filepath, write_graph=True, write_images=True,
                  histogram_freq=1)]

ner_model.train(X_train, y_train, X_dev, y_dev, cb)