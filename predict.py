# -*- coding: utf-8 -*-
# @Time    : 2019/7/20 10:04
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : predict.py
# @Software: PyCharm
import numpy as np
from model.BI_Lstm_CRF import BiLSTM_CRF
from util.trainEmb import build_word_dict
from util.readData import readDataTxt
from preData.predata import changeSequence
def get_index2label():
    index2label = dict()
    idx = 0
    for l in ['o', 'a-B', 'a-I', 'b-B', 'b-I', 'c-B', 'c-I']:
        index2label[idx] = l
        idx += 1
    return index2label
def get_w2index(path):
    # 读取向量文件
    w2v = build_word_dict(path)
    # 生成词表词典
    w2index = {k: i for i, k in enumerate(sorted(w2v.keys()), 1)}
    n_vocab = len(w2v) + 1
    n_embeeding = 300
    return w2index
def get_y_origin(y_data,index2label):
    """"""
    n_sample = y_pred.shape[0]
    pred_list = []
    for i in range(n_sample):
        pred_label = [index2label[idx] for idx in np.argmax(y_data[i], axis=1)]
        pred_list.append(pred_label)
    return pred_list
if __name__ == '__main__':
    """"""
    #向量矩阵
    char_embedding_mat = np.load('datagrand/embedding_matrix.npy')
    # #
    # w2index = get_w2index('../datagrand/w2v/grand.w2v.300d.txt')
    # #
    # index2w = {i : w for w,i in w2index.items()}
    x_test,_ = changeSequence(readDataTxt('../datagrand/test.txt'))
    print(np.shape(x_test))
    #读取测试集
    X_test = np.load('datagrand/test_x.npy')
    ner_model = BiLSTM_CRF(n_input=200, n_vocab=char_embedding_mat.shape[0],
                           n_embed=300, embeding_mat=char_embedding_mat,
                           keep_prob=0.5, n_lstm=500, keep_prob_lstm=0.6,
                           n_entity=7, optimizer='adam', batch_size=32, epochs=500)
    model_file = 'datagrand/bilstm_crf_best.hdf5'
    ner_model.model.load_weights(model_file)
    #预测模型
    y_pred = ner_model.model.predict(X_test[:, :])
    print(y_pred.shape)
    pred_list = get_y_origin(y_pred,get_index2label())
    print(np.shape(pred_list))
    

