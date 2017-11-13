#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from datetime import datetime,timedelta
import pandas as pd
import os,copy,math,time
import numpy as np
np.random.seed(2017)
from sklearn import preprocessing
from sklearn.utils import shuffle

from config import config
from utils import *

# 划分训练数据集合测试数据集
def split_train_test():
    action = pd.read_csv(config.action_data_file,sep="\t")
    split_date = datetime.strptime(config.split_date, config.date_format )
    fun = lambda x: datetime.strptime(x,config.date_format)
    action["FirstDate"] = action["FirstDate"].map( fun )
    train_df = action[ action["FirstDate"]<split_date ].reset_index(drop=True)
    train_df["FirstDate"] = train_df["FirstDate"].map( lambda x: x.strftime(config.date_format) )
    train_df.to_csv(config.train_data_file,index=False,index_label=False)
    
    test_df = action[ action["FirstDate"]>=split_date ].reset_index(drop=True)
    test_df["FirstDate"] = test_df["FirstDate"].map( lambda x: x.strftime(config.date_format) )
    test_df.to_csv(config.test_data_file,index=False,index_label=False)

# 获取label=1的用户
def get_label_1_users(action):
    user_df = action[["CustomerID","label"]].groupby(["CustomerID"],as_index=False).sum().reset_index(drop=True)
    user_df = user_df[user_df["label"]>=1].reset_index(drop=True)
    user_df["label"] = 1.0
    return user_df 

# groupby CustomerID 前操作
def gen_feat1( data_file ):
    print( "begin to gen_feat1 ......" )
    action = pd.read_csv(data_file)
    # 去除掉退货的数据: Trans=-1的列
    action = action[action["Trans"]==1].reset_index(drop=True)
    fun1 = lambda x: datetime.strptime(x, config.date_format) - datetime.strptime(config.begin_date, config.date_format)
    fun2 = lambda x: fun1(x).days #时间差转换成距离历史开始的天数
    action["FirstDate"] = action["FirstDate"].map( fun2 )
    action["OrderDate"] = action["OrderDate"].map( fun2 )
    # 标注label
    action["label"] = 0.0
    label_1_idx = action[ (action["OrderDate"]-action["FirstDate"]<=45)&(action["OrderDate"]-action["FirstDate"]>=1)].index
    action.ix[list(label_1_idx),"label"] = 1.0
    user_df = get_label_1_users(action)
    del action["label"]
    action = pd.merge(action,user_df,on="CustomerID",how="left")
    action = action.replace(np.nan,0.0)
    # 获取用户age
    fun1 = lambda x: datetime.strptime(config.now_date, config.date_format) - datetime.strptime(x, config.date_format)
    fun2 = lambda x: int(fun1(x).days/365) # 转换为用户的age属性
    action["Birthday"] = action["Birthday"].map( fun2 )
    action.rename(columns={'Birthday' : 'age'}, inplace=True)
    ################## ProductCategory 列one-hot处理 ###################
    category_df = pd.get_dummies(action["ProductCategory"], prefix="")
    action = pd.concat([action,category_df],axis=1)
    # 转换product为int值
    action["Product"] = action["Product"].map( lambda x: int(x[3:]) )
    del action["ProductCategory"],action["FirstDate"],action["OrderDate"],action["Trans"]
    action = action.groupby(["CustomerID"],as_index=False).first().reset_index(drop=True)
    return action

# 获得train_data或者vali_data
def get_train_data():
    print( "begin to get_train_data ..." )
    user_df = gen_feat1(config.train_data_file)
    user_df = shuffle( user_df )
    train_column = ["age","Product","Items","UnitPrice","_Category1","_Category2","_Category3"]
    train_x = user_df[train_column].values
    train_y = user_df["label"].values
    return train_x, train_y

def get_test_data():
    print( "begin to get_test_data ..." )
    user_df = gen_feat1(config.test_data_file)
    user_df = user_df[user_df["label"]==0].reset_index(drop=0)
    test_column = ["age","Product","Items","UnitPrice","_Category1","_Category2","_Category3"]
    test_x = user_df[test_column].values
    user_id = user_df["CustomerID"].values
    return user_id, test_x

if __name__ == '__main__':
    split_train_test()
    # action = pd.read_csv(config.action_data_file,sep="\t")
    # a = action[["CustomerID"]].drop_duplicates(["CustomerID"])
    # print len(a.index)
    # split_train_test()
    # train_x, train_y = get_train_data()
    # user_id, test_x = get_test_data()

