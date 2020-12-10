# Baseline models for answering the question correctly
# Author: Shaurya

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import itertools
from collections import Counter
from dask.distributed import Client,progress
from dask_ml.wrappers import ParallelPostFit
from sklearn.model_selection import train_test_split
from dask_ml.model_selection import train_test_split as dask_split
import dask
import dask_xgboost

#? Dask client and workers
client=Client(n_workers=4,threads_per_worker=2,memory_limit='2GB')
client

# Read in the train
train=dd.read_parquet("data/train.parquet")
train.head()

# Questions and lectures metadata
questions=dd.read_csv("data/questions.csv")
lectures=dd.read_csv("data/lectures.csv")

# Columns
train.columns
questions.columns
lectures.columns

# Run dataset raw on train
# might need to standardscaler and remove nulls
train['answered_correctly'].value_counts().compute()
# remove -1s for lectures
train=train[train['answered_correctly']!=-1]
train=train.dropna()

X=train[['timestamp','user_id','content_id','content_type_id','task_container_id','user_answer','prior_question_elapsed_time','prior_question_had_explanation']]
y=train['answered_correctly']
x_train,x_test,y_train,y_test=dask_split(X,y,test_size=0.2)

x_train.shape
y_train.shape

# xgboost params
params={
    'objective':'binary:logistic',
    'max_depth':4,
    'eta':0.01,
    'subsample':0.5,
    'min_child_weight':0.5
}

result=dask_xgboost.train(client,params,x_train,y_train,num_boost_round=10)
