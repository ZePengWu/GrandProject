# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 16:54
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : predata.py
# @Software: PyCharm

from util.readData import readDataTxt
import numpy as np

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

        print(len(x),len(y))
        print(y)
        data_x.append(x)
        data_y.append(y)
    print(max_count,count)
    return data_x,data_y
def readVector(path):
    """
    返回 index2word,word2index,embeding_mat
    :param path:
    :return:
    """
    v2e = {}

    return v2e
if __name__ == '__main__':

    test_data = readDataTxt('../datagrand/test.txt')
    test_x,test_y = changeSequence(test_data)
    train_data = readDataTxt('../datagrand/train.txt')
    train_x_all, train_y_all = changeSequence(train_data)
    # print(np.shape(train_x_all))
    # print(np.shape(train_y_all))

    train_x, dev_x = train_x_all[:15000], train_x_all[15000:]
    train_y, dev_y = train_y_all[:15000], train_y_all[15000:]
    print(type(train_x))
    print(np.array(train_x).shape,np.shape(train_y))
    print(np.shape(dev_x),np.shape(dev_y))
    print(np.shape(test_x),np.shape(test_y))

