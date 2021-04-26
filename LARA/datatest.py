import pandas as pd
import numpy as np
import torch
from torch.utils import data

attr_num=18  #属性的个数
attr_present_dim=5
train=np.array(pd.read_csv("./train_data.csv"))
batch=train[8:80]
user=[x[0] for x in batch]
attr=[x[2][1:-1].split() for x in batch]
c=np.int32(attr)
user_emb_matrix=np.array(pd.read_csv('util/user_emb.csv',header=None))
attr_matrix=torch.randn(2*attr_num,attr_present_dim,requires_grad=True).float()
class TrainDataset(data.Dataset):
    def __init__(self):
        self.data=np.array(pd.read_csv("./train_data.csv"))
        self.emb=user_emb_matrix
    def __getitem__(self, index):
        train_data=self.data[index]
        userid=train_data[0]
        attr=np.int32(train_data[2][1:-1].split())
        attr=torch.from_numpy(attr).long()
        user_emb=self.emb[userid]
        user_emb=torch.from_numpy(user_emb).float()
        return attr,user_emb
    def __len__(self):
        return len(self.data)
train=TrainDataset()
train_loader=data.DataLoader(train,shuffle=True,batch_size=128,num_workers=2)
result=torch.empty(128,108)
for data in train_loader:
    attr,user=data
    a=torch.ones(len(attr),108)
    for i in range(len(attr)):
        attr_emb=torch.index_select(attr_matrix,0,attr[i]).reshape(-1,attr_num*attr_present_dim)
        ur=user[i].reshape(1,18)
        user_emb=torch.cat((attr_emb,ur),1)
        a[i]=user_emb
    print(a.shape)
    print(a)
    break












