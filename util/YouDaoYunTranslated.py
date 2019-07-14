# -*- coding: utf-8 -*-
# @Time    : 2019/7/7 20:52
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    :
# @Software: PyCharm

import uuid
import requests
import hashlib
import time
import json

YOUDAO_URL = 'http://openapi.youdao.com/api'
APP_KEY = '10af0afdcc806dae'
APP_SECRET = 'kku9nIbpZl6zBAZ0jG6KrSptMKYSZF1A'

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)

def connect(text,rLanguage,tLanguage):
    """
    EN
    zh-CHS
    :return:
    """
    #需要翻译的文本
    q = text

    data = {}
    #源语言
    data['from'] = rLanguage
    #目标语言
    data['to'] = tLanguage
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign

    response = do_request(data)
    content = response.content
    content = bytes.decode(content)
    translateStr = json.loads(content)['translation'][0]
    # print (transleateStr)
    # print(type(content))
    return translateStr

if __name__ == '__main__':
    translateStr_English = connect('选择身高范围155-165，因为人数集中，身高相差不大。','zh-CHS','EN')
    translateStr_Chinese = connect(translateStr_English,"EN",'zh-CHS')
    print('选择身高范围155-165，因为人数集中，身高相差不大。')
    print(translateStr_English)
    print(translateStr_Chinese)