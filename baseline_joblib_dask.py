# Baseline for riiid with joblib backend for sklearn
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
import joblib
from sklearn.linear_model import LogisticRegression

client=Client(n_workers=4,threads_per_worker=2,memory_limit='4GB')
client

# read in with pandas
dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}
train=dd.read_parquet("data/train.parquet",dtype=dtypes)

# Run dataset raw on train
# might need to standardscaler and remove nulls
train['answered_correctly'].value_counts().compute()
# remove -1s for lectures
train=train[train['answered_correctly']!=-1]
train=train.dropna()

X=train[['timestamp','user_id','content_id','content_type_id','task_container_id','user_answer','prior_question_elapsed_time','prior_question_had_explanation']]
y=train['answered_correctly']
x_train,x_test,y_train,y_test=dask_split(X,y,test_size=0.2)

# classifier
clf=LogisticRegression()
with joblib.parallel_backend('dask'):
    clf.fit(x_train,y_train)






















