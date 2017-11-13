#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os,copy,math,time
import numpy as np

from sklearn.ensemble import RandomForestClassifier  
from sklearn import preprocessing
from sklearn.utils import shuffle

from config import config
from utils import *
from gen_feat import get_train_data, get_test_data

train_x, train_y = get_train_data()
# print train_x[:10]
print( len(train_y),np.sum(train_y) ) # 打印训练样本个数和 正样本个数
user_id, test_x = get_test_data()
print( len(user_id) ) # 打印测试样本个数

model = RandomForestClassifier(n_estimators=800)  
model.fit( train_x, train_y )
pred = model.predict( test_x )

# 构造结果文件
ans = pd.DataFrame(user_id,columns=["user_id"])
ans["pred"] = pred
ans = ans[ans["pred"]==1].reset_index(drop=True)
ans = ans.sort_values("user_id")
ans.to_csv(config.ans_file,index=False,index_label=False)


