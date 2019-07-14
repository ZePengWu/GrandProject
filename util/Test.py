# -*- coding: utf-8 -*-
# @Time    : 2019/7/7 19:01
# @Author  : Ze-peng Wu(午泽鹏)
# @Email   : wuzepeng_sxu@126.com
# @File    : Test.py.py
# @Software: PyCharm

# coding:utf-8
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkalimt.request.v20190107 import TranslateECommerceRequest
import json


def TranslatedZHtoEN(text,rLanguage,tLanguage):
    """
    zh,en
    :param text:
    :param rLanguage:
    :param tLanguage:
    :return:
    """
    # 创建AcsClient实例
    client = AcsClient(
        "LTAIS3QsrWpDoHGg",
    "sghwZ70A26aqx8MH6gK5T7WVrC00k0",
    "cn-hangzhou" # 地域ID
    )
    # 创建request，并设置参数
    request = TranslateECommerceRequest.TranslateECommerceRequest()
    request.set_SourceLanguage(rLanguage) # 源语言
    request.set_Scene("title") #设置场景，商品标题: title，商品描述: description，商品沟通: communication
    request.set_SourceText(text) #原文
    request.set_FormatType("text") #翻译文本的格式
    request.set_TargetLanguage(tLanguage) #目标语言
    # 发起API请求并显示返回值
    response = client.do_action_with_exception(request)
    response = bytes.decode(response)
    response = json.loads(response)
    # print(response['Data']['Translated'])
    # print(type(response))
    return response['Data']['Translated']

if __name__ == '__main__':
    enlishStr = TranslatedZHtoEN('因为该范围内人数集中，身高差距很小。', 'zh', 'en')
    zhStr = TranslatedZHtoEN(enlishStr,'en','zh')
    print('选择身高范围155-165，因为人数集中，身高相差不大。')
    print(enlishStr)
    print(zhStr)