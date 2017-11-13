#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os,copy,math,time,h5py
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
#from svmutil import svm_read_problem

def one_hot( data_ary, one_hot_len):
    # data_ary = array([1,2,3,5,6,7,9])
    # one_hot_len: one_hot最长列
    max_num = np.max(data_ary);
    ans = np.zeros((len(data_ary),one_hot_len),dtype=np.float64)
    for i in range(len(data_ary)):
        ans[ i, int(data_ary[i]) ] = 1.0
    return ans

def re_onehot( data_ary ):
    # data_ary = array([[0,0,0,1.0],[1.0,0,0,0],...])
    ans = np.zeros((len(data_ary),),dtype=np.float64)
    for i in range(len(data_ary)):
        for j in range(len(data_ary[i,:])):
            if data_ary[i,j] == 1.0:
                ans[i] = 1.0*j;
                break;
    return ans

# 将数据写入h5文件
def write2H5(h5DumpFile,data):
    # if not os.path.exists(h5DumpFile): os.makedir(h5DumpFile)
    with h5py.File(h5DumpFile, "w") as f:
        f.create_dataset("data", data=data, dtype=np.float64)

# 将数据从h5文件导出
def readH5(h5DumpFile):
    feat = [];
    with h5py.File(h5DumpFile, "r") as f:
        feat.append(f['data'][:])
    feat = np.concatenate(feat, 1)
    print('readH5 Feature.shape=', feat.shape)
    return feat.astype(np.float64)

# 一般矩阵归一化
def my_normalization( data_ary, scope=(-1.0,1.0), axis=0 ):
    # axis = 0 按列归一化; 1时按行归一化
    if axis == 1:
        data_ary = np.matrix(data_ary).T
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=scope)
        ans = min_max_scaler.fit_transform(ans)
        ans = np.matrix(ans).T
        ans = np.array(ans)
    else:
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=scope)
        ans = min_max_scaler.fit_transform(ans)
    return ans












