#-*- coding:utf-8 -*-
import os
import numpy as np
from easydict import EasyDict as edict
if not os.path.exists("./cache/"): os.makedirs("./cache/")

config = edict()
config.action_data_file = "./data/action_data.csv"
config.train_data_file = "./cache/train.csv" # 训练数据文件目录
config.test_data_file  = "./cache/test.csv" # 测试数据文件目录
config.ans_file = "./cache/ans.csv" # 最终模型预测生成文件目录

config.now_date = "2017/6/30" # 计算用户年龄用
config.begin_date = "2016/12/31" # 历史开始的日期
config.split_date = "2017/5/20" # 测试集切分点日期
config.date_format = "%Y/%m/%d"




