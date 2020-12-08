# Initial EDA for Riiid Education Challenge
# Author: Shaurya

import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
from dask.distributed import Client,progress
from datetime import datetime
import itertools

client=Client(n_workers=4,threads_per_worker=2,memory_limit='2GB')
client
# Read in train parquet
train=dd.read_parquet("data/train.parquet")
train.info()
train.head()

# read in questions metadata
questions=dd.read_csv("data/questions.csv")
questions.info()

# read in lectures metadata
lectures=dd.read_csv("data/lectures.csv")
lectures.info()

#! Explore train.csv
#? target: answered_correctly
correct_counts=train['answered_correctly'].value_counts()
plt.figure(figsize=(15,10))
correct_counts.compute().plot(kind='bar')
plt.title("Target varible: Answered correctly")

#? timestamp: time in milliseconds between user interaction and first event completion
# how does it affect if the answer was correctly answered
correct_timestamp=train.groupby(['answered_correctly']).agg({'timestamp':['mean','std']}).apply(lambda x:x/1000)
correct_timestamp_computed=correct_timestamp.compute()
overall_avg_timestamp=train['timestamp'].mean().compute()/1000

#? user_id: ID code for the student
# unique users
len(train['user_id'].unique().compute()) # 393,656 
# average number of records per student; number of questions on average
train.groupby(['user_id']).count().mean().compute() # 257 questions

#? content_id: ID code for interaction; question_id from questions
# unique questions
train['content_id'].nunique().compute() # 13,782 unique questions
# for each question, how many do we have answered correctly
correct_question=train.groupby(['content_id','answered_correctly'])['row_id'].count().compute()
correct_question=pd.DataFrame(correct_question)
correct_question.info()
correct_question=correct_question.reset_index()
correct_question=correct_question[correct_question['row_id']>5000]
fig1=go.Figure(data=[
    go.Bar(name='Wrong',x=correct_question[correct_question['answered_correctly']==0]['content_id'],y=correct_question[correct_question['answered_correctly']==0]['row_id']),
    go.Bar(name='Right',x=correct_question[correct_question['answered_correctly']==1]['content_id'],y=correct_question[correct_question['answered_correctly']==1]['row_id'])
])
fig1.update_layout(barmode='group')
fig1.show()

#? content_type_id: ID for type of interaction; question or lecture
# number of questions to lectures
plt.figure(figsize=(15,10))
train['content_type_id'].value_counts().compute().plot(kind='bar')
plt.xlabel("Count")
plt.ylabel("Content type")
plt.title("Count of content type: Question or Lecture")
# most are questions, very few lectures

#? task_container_id: ID for batch of questions/lectures
# number of task container ids
train['task_container_id'].nunique().compute() # 10,000
# number of times a task container is answered
task_containers=train.groupby(['task_container_id'])['task_container_id'].count().compute()
task_containers=task_containers.sort_values(ascending=False)
plt.figure(figsize=(15,10))
task_containers[:10].plot(kind='bar')
plt.title("Num. times task containers answered")
# avg number of questions in each task_container_id
#! train.groupby(['task_container_id'])['content_id'].nunique().compute()

#? user_answer: the user's answer to the question
# can be verified with correct_answer

#? prior question had explanation: bool 
# number of times user saw explanation
plt.figure(figsize=(15,10))
train['prior_question_had_explanation'].value_counts().compute().plot(kind='bar')
plt.title("Num. times user saw explanation before question")

#! Exploring questions.csv
questions.columns
questions.head()

#? question_id
# total unique questions
questions['question_id'].nunique().compute() # 13,523

#? bundle_id
# total bundle ids; unique
questions['bundle_id'].nunique().compute() # 9,765

#? How many questions per bundle?
bundle_questions=questions.groupby(['bundle_id'])['question_id'].nunique().compute()
bundle_questions=bundle_questions.sort_values(ascending=False)
bundle_questions[:10].plot(kind='bar')
bundle_questions.describe() # max:5, min:1

#? part; the data description says relevant part of the TOEIC test in South Korea
# how many parts are there in the test
questions['part'].nunique().compute() # 7 parts
# how many questions per part do we have
questions.groupby(['part']).agg({'question_id':'count'}).compute()
# 	question_id
# part	
# 1	992
# 2	1647
# 3	1562
# 4	1439
# 5	5511
# 6	1212
# 7	1160

# Part 5 has the most questions at 5.5k
# rest are somewhat equal"ish" distributed

#? Tags: topics for the questions
# anonymised tags for topics
# check how many unique tags
tags=questions['tags']
tags=pd.DataFrame(tags)
tags=tags.dropna()
ntags=[]
for item in tags[0]:
    row=item.split(' ')
    ntags.append(row)
ntags=list(itertools.chain.from_iterable(ntags))
ntags=list(map(lambda x:int(x),ntags))
plt.figure(figsize=(15,10))
pd.Series(ntags).value_counts().plot(kind='bar')
plt.title("Count of tags")
count_ntags=pd.Series(ntags).value_counts()
plt.boxplot(count_ntags)