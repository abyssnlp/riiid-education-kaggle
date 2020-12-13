# SAKT: Self attention for knowledge tracing
# Author: Shaurya

import numpy as np
import pandas as pd

import os
for dirname,_,filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname,filename))

import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

print(torch.cuda.is_available())

max_seq=160

# Load the data
# only subset of the data
dtype={
    'timestamp':'int64',
    'user_id':'int32',
    'content_id':'int16',
    'content_type_id':'int8',
    'answered_correctly':'int8'
}

train=pd.read_csv('data/train.csv',usecols=[1,2,3,4,7],dtype=dtype)
train.head()

# pre-process
pd.value_counts(train['content_type_id'])
# remove the answers in lectures (only questions)
train=train[train['content_type_id']==False]
gc.collect()

# arrange data by timestamp ascending
train=train.sort_values(['timestamp'],ascending=True).reset_index(drop=True)
# skills (content id)
skills=train['content_id'].unique()
len(skills) # 13523 type of questions

# further select only user_id(user), content_id(question) and answered_correctly(target)
#? For each user, features: questions answered(over time) and answered_correctly(for each question)
group=train[['user_id','content_id','answered_correctly']].groupby(['user_id']).apply(
    lambda r:(
        r['content_id'].values,
        r['answered_correctly'].values
    )
)
del train
gc.collect()

# random seed
random.seed(1)

# Selft attention knowledge tracing algorithm
# embedding for exercise and answers for user with attention

class SAKTDataset(Dataset):
    def __init__(self,group,n_skill,max_seq=max_seq):
        super(SAKTDataset,self).__init__()
        self.max_seq=max_seq
        self.n_skill=n_skill
        self.samples=group

        self.user_ids=[]
        # group index is user_id from groupby on train
        for user_id in group.index:
            q,qa=group[user_id]
            if len(q)<2:
                continue
            self.user_ids.append(user_id)
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self,index):
        user_id=self.user_ids[index]
        q_,qa_=self.samples[user_id]
        seq_len=len(q_)
        q=np.zeros(self.max_seq,dtype=int)
        qa=np.zeros(self.max_seq,dtype=int)

        if seq_len>=self.max_seq:
            if random.random()>0.1:
                start=random.randint(0,(seq_len-self.max_seq))
                end=start+self.max_seq
                q[:]=q_[start:end]
                qa[:]=qa_[start:end]
            else:
                q[:]=q_[-self.max_seq:]
                qa[:]=qa_[-self.max_seq]
        else:
            if random.random()>0.1:
                start=0
                end=random.randint(2,seq_len)
                seq_len=end-start
                q[-seq_len:]=q_[0:seq_len]
                qa[-seq_len:]=qa_[0:seq_len]
            else:
                q[-seq_len:]=q_
                qa[-seq_len:]=qa_
        target_id=q[1:]
        label=qa[1:]
        x=np.zeros(self.max_seq-1,dtype=int)
        x=q[:-1].copy()
        x+=(qa[:-1]==1)*self.n_skill
        return x,target_id,label

dataset=SAKTDataset(group,len(skills))
dataloader=DataLoader(dataset,batch_size=2048,shuffle=True,num_workers=0)

item=dataset.__getitem__(5)

# Model Definition
class feedforwardNet(nn.Module):
    def __init__(self,state_size=200):
        super(feedforwardNet,self).__init__()
        self.state_size=state_size
        self.lr1=nn.Linear(state_size,state_size)
        self.relu=nn.ReLU()
        self.lr2=nn.Linear(state_size,state_size)
        self.dropout=nn.Dropout(0.2)
    
    def forward(self,x):
        x=self.lr1(x)
        x=self.relu(x)
        x=self.lr2(x)
        return self.dropout(x)
    
# test
arr=np.array([1,2,4,5,5])
torch.from_numpy(arr)

def future_mask(seq_length):
    future_mask=np.triu(np.ones((seq_length,seq_length)),k=1).astype('bool')
    return torch.from_numpy(future_mask)

class SAKTModel(nn.Module):
    def __init__(self,n_skill,max_seq=max_seq,embed_dim=128):
        super(SAKTModel,self).__init__()
        self.n_skill=n_skill
        self.embed_dim=embed_dim
        self.embedding=nn.Embedding(2*n_skill+1,embed_dim)
        self.pos_embedding=nn.Embedding(max_seq-1,embed_dim)
        self.e_embedding=nn.Embedding(n_skill+1,embed_dim)
        self.multi_att=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=8,dropout=0.2)
        self.dropout=nn.Dropout(0.2)
        self.layer_normal=nn.LayerNorm(embed_dim)
        self.ffn=feedforwardNet(embed_dim)
        self.pred=nn.Linear(embed_dim,1)
    # forward pass over each question
    def forward(self,x,question_ids):
        device=x.device
        x=self.embedding(x)
        pos_id=torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x=self.pos_embedding(pos_id)
        x=x+pos_x
        e=self.e_embedding(question_ids)
        x=x.permute(1,0,2)
        e=e.permute(1,0,2)
        att_mask=future_mask(x.size(0)).to(device)
        att_output,att_weight=self.multi_att(e,x,x,attn_mask=att_mask)
        att_output=self.layer_normal(att_output+e)
        att_output=att_output.permute(1,0,2)
        x=self.ffn(att_output)
        x=self.layer_normal(x+att_output)
        x=self.pred(x)
        return x.squeeze(-1),att_weight

device="cuda"
model=SAKTModel(len(skills),embed_dim=128)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
criterion=nn.BCEWithLogitsLoss()
model.to(device)
criterion.to(device)

# Training the model
def train_epoch(model,train_iterator,optim,criterion,device='cuda'):
    model.train()
    train_loss=[]
    num_corrects=[]
    num_total=[]
    labels=[]
    outs=[]

    tbar=tqdm(train_iterator)
    for item in tbar:
        x=item[0].to(device).long()
        target_id=item[1].to(device).long()
        label=item[2].to(device).float()

        optim.zero_grad()
        output,attn_weight=model(x,target_id)
        loss=criterion(output,label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        output=output[:,-1]
        label=label[:,-1]
        pred=(torch.sigmoid(output)>=0.5).long()
        num_corrects+=(pred==label).sum().item()
        num_total+=len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())
        tbar.set_description('loss - {:.4f}'.format(loss))

    acc=num_corrects/num_total
    auc=roc_auc_score(labels,outs)
    loss=np.mean(train_loss)
    return loss,acc,auc

torch.cuda.empty_cache()
# Train
epochs=40
for epoch in range(epochs):
    loss,acc,auc=train_epoch(model,dataloader,optimizer,criterion,device)
    print("epoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, loss, acc, auc))

#! CUDA out of memory errors
# try running on Kaggle