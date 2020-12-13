# Baseline model with Pandas
# Author: Shaurya

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import gc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score

# read in the train, questions and lectures
train=pd.read_parquet("data/train.parquet")
questions=pd.read_csv("data/questions.csv")
lectures=pd.read_csv("data/lectures.csv")

# train data
train.head()
train.info()

# train using basic logistic regression
# no feature engineering
train=train[train['answered_correctly']!=-1]
labels=train['answered_correctly']
train=train[['timestamp','user_id','content_id','content_type_id','task_container_id']]
gc.collect()

# nulls
train.shape
train=train.dropna()

# train and test split
x_train,x_test,y_train,y_test=train_test_split(train,labels,test_size=0.2)
# del train
clf=LogisticRegression(n_jobs=-1)
clf.fit(x_train,y_train)

# predict on the test set
y_pred=clf.predict(x_test)
print(accuracy_score(y_pred,y_test)) # 65.7%
print(roc_auc_score(y_test,y_pred)) 

#! Not the best approach due to class imbalance
# weighted regression with grid search for optimal weights for 0 and 1























