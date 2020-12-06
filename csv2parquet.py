# Converting CSV to Parquet for fast reads on subsequent efforts to tackle the challenge
# Author: Shaurya

import pandas as pd
import dask.dataframe as dd
import os
import math
import time

# check files and filesizes (in MB)
for item in os.listdir('data'):
    print(item,end=' =: ')
    print('%.2f'%(os.path.getsize(os.path.join('data',item))/(1024*1024)))


# Read in the dataframe
train=dd.read_csv("data/train.csv")

# save as parquet ( dask chunks it in 20MB parquet chunks )
train.to_parquet("data/train.parquet",engine='pyarrow')

# Check read speed
start=time.time()
train=dd.read_parquet("data/train.parquet",engine='pyarrow')
end=time.time()
print("Train parquet read time: %s"%(end-start))