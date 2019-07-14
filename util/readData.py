# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 12:18
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : readData.py
# @Software: PyCharm

import json

def readDataTxt(path):
    """
    读取数据
    :param path:
    :return:
    """
    data = []
    f = open(path,'r',encoding="utf-8")
    for line in f:
        d = str(line).rstrip('\n')
        data.append(d)
    return data
def writeDataTxt(path,data):
    """
    写入数据
    :param path:
    :param data:
    :return:
    """
    f = open(path, 'a', encoding="utf-8")
    for line in data:
        lineStr = str(line).replace("_"," ")
        print(lineStr)
        f.write(lineStr + "\n")
    print("数据写入完成...")
def countOfWord(data):
    w = {}
    for line in data:
        lineStr = str(line).split("_")
        for word in lineStr:
            w[word] = 0

    return len(w)


if __name__ == '__main__':
    data = readDataTxt('../datagrand/corpus.txt')
    # writeDataTxt('../datagrand/newCropus.txt',data)
    print(countOfWord(data))