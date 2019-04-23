#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: test.py
@time: 2019-04-23 16:05
@description:
"""


def split_type(x):
    """
    分割房屋类型
    :param x:
    :return:
    """
    assert len(x) == 6, "x的长度必须为6"
    return int(x[0]),int(x[2]),int(x[4])

res=split_type("2室2厅2卫")
print(res)


