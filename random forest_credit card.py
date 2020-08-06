# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:08:01 2018

@author: LENOVO S435
"""

import pandas as pd

#读取数据
credit=pd.read_excel('E:\\study\\postgraduate projects\\random forest credit card\\DATA\\q2.xlsx')
'''
X=credit[['gender','marriage','education','company','job','age','private_company',
'MAX_WHITEGOLD','MAX_CREDIT','normalcards_num','cards_num','avgasset_3','loan',
'debt','avgdeposit_3','asset']]
'''
#空值处理
credit= credit.fillna(credit.mean())

#特征集选择
X=credit[['card_num','gender','marriage','education','company','job','age','private_company',
'MAX_WHITEGOLD','MAX_CREDIT','cards_num','avgasset_3','loan',
'debt','avgdeposit_3','asset']]

#被预测变量
y=credit['STATUS']

from sklearn.model_selection import train_test_split

#构建训练集与预测集
X_train0,X_test0,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=33)

X_train=X_train0[['gender','marriage','education','company','job','age','private_company',
'MAX_WHITEGOLD','MAX_CREDIT','cards_num','avgasset_3','loan',
'debt','avgdeposit_3','asset']]
X_test=X_test0[['gender','marriage','education','company','job','age','private_company',
'MAX_WHITEGOLD','MAX_CREDIT','cards_num','avgasset_3','loan',
'debt','avgdeposit_3','asset']]

#数据格式转化
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

#使用随机森林进行预测
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)

#使用决策树进行预测
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred=dtc.predict(X_test)

#使用K近邻算法进行预测
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
knc_y_pred=knc.predict(X_test)

#使用逻辑回归进行预测
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr_y_pred=lr.predict(X_test)

#随机森林预测性能分析
from sklearn.metrics import classification_report
print('The accuracy of random forest classifier is',rfc.score(X_test,y_test))
print('The accuracy of random forest classifier is',rfc.score(X_train,y_train))
print (classification_report(rfc_y_pred,y_test))

#决策树预测性能分析
print('The accuracy of DecisionTree is',dtc.score(X_test,y_test))
print('The accuracy of DecisionTree is',dtc.score(X_train,y_train))
print (classification_report(dtc_y_pred,y_test))

#K近邻算法性能分析
print('The accuracy of KNN is',knc.score(X_test,y_test))
print('The accuracy of KNN is',knc.score(X_train,y_train))
print (classification_report(knc_y_pred,y_test))

#逻辑回归预测性能分析
print('The accuracy of LogisticRegression is',lr.score(X_test,y_test))
print('The accuracy of LogisticRegression is',lr.score(X_train,y_train))
print (classification_report(lr_y_pred,y_test))

rfc_y_pred1=pd.DataFrame(rfc_y_pred)
X_test1=X_test0.reset_index(drop = True)
test_result=X_test1.join(rfc_y_pred1)
test_result.to_excel('E:\\study\\postgraduate projects\\random forest credit card\\DATA\\q2result.xlsx', index=False, header=False)