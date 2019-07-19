# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 16:54
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : predata.py
# @Software: PyCharm

from util.readData import readDataTxt
from util.trainEmb import build_word_dict
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import numpy as np
import os
def creatLable(substr, lable='o'):
    """"""
    L = str(substr).split("_")
    x = []
    y = []
    for word in L:
        index = int(word)
        if index == 0 :
            print(True)
        x.append(index)
        y.append(lable)
    start = 0
    # end = len(y) - 1
    if lable != 'o':
        for index in range(len(y)):
            if index == start:
                y[index] = y[index] + "-B"
            else:
                y[index] = y[index] + "-I"
    return x, y
def changeSequence(data):
    """"""
    """
    /a,/b,/c,/o
    """
    max_count = 0
    count = 0
    data_x = []
    data_y = []
    for line in data:
        #以两个作为切分
        subList = str(line).strip('\n').split("  ")
        x = []
        y = []
        for substr in subList:
            #
            sub_x,sub_y = [],[]
            if '/a' in substr:
                sub_x,sub_y = creatLable(substr.replace('/a',''),'a')
            elif '/b' in substr:
                sub_x,sub_y = creatLable(substr.replace('/b',''),'b')
            elif '/c' in substr:
                sub_x,sub_y = creatLable(substr.replace('/c',''),'c')
            else:
                sub_x, sub_y = creatLable(substr.replace('/o', ''))
            x.extend(sub_x)
            y.extend(sub_y)
        if max_count < len(x):
            max_count = len(x)
        if len(x) >= 200:
            count += 1
        # print(len(x),len(y))
        # print(y)
        data_x.append(x)
        data_y.append(y)
    # print(max_count,count)
    # print(data_x,data_y)
    return data_x,data_y
def get_embedding_mat_and_w2index(path):
    """
    返回 index2word,word2index,embeding_mat
    :param path:
    :return:
    """
    # 读取向量文件
    w2v = build_word_dict(path)
    # 生成词表词典
    w2index = { k : i for i , k in enumerate(sorted( w2v.keys() ) , 1 )}
    n_vocab = len(w2v) + 1
    n_embeeding = 300
    # 生成此嵌入矩阵 emb_mat
    embeeding_mat = np.zeros([n_vocab,n_embeeding])
    for word,index in w2index.items():
        vec = w2v.get(word)
        if vec is not None:
            embeeding_mat[index] = vec
    return embeeding_mat,w2index
def get_x_data_index(data,w2index,sequence_len):
    """
    将数据转化为 index（索引表示）
    :param data: 原数据
    :param w2index: 词典
    :param sequence_len: 最大序列长度
    :return:
    """
    index_data = []
    for l in data:
        # print(l)
        # print([w2index[str(s)] if w2index.get(str(s)) is not None else 0 for s in l])
        index_data.append([w2index[str(s)] if w2index.get(str(s)) is not None else 0 for s in l])
        # print(index_data)
    index_array = pad_sequences(index_data,maxlen = sequence_len,dtype='int32',
                                padding='post',truncating='post',value=0)

    return index_array


def get_y_data_index(data, lable2index, sequence_len):
    """
    将数据转化为 index（索引表示）
    :param data: 原数据
    :param w2index: 词典
    :param sequence_len: 最大序列长度
    :return:
    """
    index_data = []
    for l in data:
        # print(l)
        # print([lable2index[str(s)] for s in l])
        # print([lable2index[s] for s in l])

        index_data.append([lable2index[str(s)] for s in l])

    index_array = pad_sequences(index_data, maxlen=sequence_len, dtype='int32',
                                padding='post', truncating='post', value=0)
    index_array = to_categorical(index_array,num_classes=7)
    return index_array

if __name__ == '__main__':
    embbedding_mat_files = '../datagrand/embedding_matrix.npy'
    #数据添加
    test_data = readDataTxt('../datagrand/test.txt')
    test_x,test_y = changeSequence(test_data)
    train_data = readDataTxt('../datagrand/train.txt')
    train_x_all, train_y_all = changeSequence(train_data)
    #分割训练集 开发集
    train_x, dev_x = train_x_all[:15000], train_x_all[15000:]
    train_y, dev_y = train_y_all[:15000], train_y_all[15000:]
    embbedding_mat,w2index=get_embedding_mat_and_w2index('../datagrand/w2v/grand.w2v.300d.txt')
    if not os.path.exists(embbedding_mat_files):
        np.save(embbedding_mat_files,embbedding_mat)
    train_x = get_x_data_index(train_x,w2index,200)
    dev_x = get_x_data_index(dev_x,w2index,200)
    test_x = get_x_data_index(test_x,w2index,200)
    print(train_x.shape,dev_x.shape,test_x.shape)
    lable2index = dict()
    idx = 0
    for l in ['o','a-B','a-I','b-B','b-I','c-B','c-I']:
        lable2index[l] = idx
        idx += 1
    train_y = get_y_data_index(train_y,lable2index,200)
    dev_y = get_y_data_index(dev_y,lable2index,200)
    test_y = get_y_data_index(test_y,lable2index,200)
    print(train_y.shape,dev_y.shape,test_y.shape)
    np.save('../datagrand/train_x.npy',train_x)
    np.save('../datagrand/train_y.npy',train_y)
    np.save('../datagrand/test_x.npy',test_x)
    np.save('../datagrand/test_y.npy',test_y)
    np.save('../datagrand/dev_x.npy',dev_x)
    np.save('../datagrand/dev_y.npy',dev_y)



