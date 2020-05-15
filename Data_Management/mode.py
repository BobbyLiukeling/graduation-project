# -*- encoding: utf-8 -*-
# @author : bobby
# @time : 2020/2/9 14:20

#载入数据 、查看相关信息
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pdb

#第一步：加载、查看数据
band_data = pd.read_csv('my_dict.csv',encoding='UTF-8')
band_data.info()
band_data.shape


# 第二步：清洗、处理数据，某些数据可以使用数据库处理数据代替

#数据清洗:缺失值处理：丢去、
#查看缺失值
band_data.isnull().sum
band_data = band_data.dropna() #将空值去掉
# 去除空格
band_data['comment'] = band_data['comment'].map(lambda x: x.strip())

for column in band_data.columns:
    le = LabelEncoder()
    band_data[column] = le.fit_transform(band_data[column])

# 模型  [重复、调优]
#'第三步：选择、训练模型'

x = band_data['sentiment_score']
y = band_data['comment']

from sklearn import model_selection
train,test,t_train,t_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=2)
model.fit(train,t_train)
pdb.set_trace()

fea_res = pd.DataFrame(x.columns,columns=['features'])
fea_res['importance'] = model.feature_importances_

t_name= band_data['churned'].value_counts()
t_name.index

import graphviz

import os
os.environ["PATH"] += os.pathsep + r'D:\software\developmentEnvironment\graphviz-2.38\release\bin'

dot_data= tree.export_graphviz(model,out_file=None,feature_names=x.columns,max_depth=2,
                         class_names=t_name.index.astype(str),
                         filled=True, rounded=True,
                         special_characters=False)
graph = graphviz.Source(dot_data)
#graph
graph.render("dtr")

#%%
print('第四步：查看、分析模型')

#结果预测
res = model.predict(test)

#混淆矩阵
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(t_test,res)
print(confmat)

#分类指标 https://blog.csdn.net/akadiao/article/details/78788864
from sklearn.metrics import classification_report
print(classification_report(t_test,res))

#%%
print('第五步：保存模型')

from sklearn.externals import joblib
joblib.dump(model,r'D:\train\201905data\mymodel.model')

#%%
print('第六步：加载新数据、使用模型')
file_path_do = r'D:\train\201905data\do_liwang.csv'

deal_data = pd.read_csv(file_path_do,encoding='UTF-8')

#数据清洗:缺失值处理

deal_data = deal_data.dropna()
deal_data['voice_mail_plan'] = deal_data['voice_mail_plan'].map(lambda x: x.strip())
deal_data['intl_plan'] = deal_data['intl_plan'].map(lambda x: x.strip())
deal_data['churned'] = deal_data['churned'].map(lambda x: x.strip())
deal_data['voice_mail_plan'] = deal_data['voice_mail_plan'].map({'no':0, 'yes':1})
deal_data.intl_plan = deal_data.intl_plan.map({'no':0, 'yes':1})

for column in deal_data.columns:
    if deal_data[column].dtype == type(object):
        le = LabelEncoder()
        deal_data[column] = le.fit_transform(deal_data[column])
#数据清洗

#加载模型
model_file_path = r'D:\train\201905data\mymodel.model'
deal_model = joblib.load(model_file_path)
#预测
res = deal_model.predict(deal_data.drop(['churned'],axis=1))

#%%
print('第七步：执行模型，提供数据')
result_file_path = r'D:\train\201905data\result_liwang.csv'

deal_data.insert(1,'pre_result',res)
deal_data[['state','pre_result']].to_csv(result_file_path,sep=',',index=True,encoding='UTF-8')