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

# Prepare dataset for modeling
train['timestamp'].describe().compute()
train['timestamp_bin']=pd.qcut(train['timestamp'],4,labels=[1,2,3,4])
train['timestamp_bin']=dd.map_partitions(pd.qcut(train['timestamp'],4,labels=[1,2,3,4]))
