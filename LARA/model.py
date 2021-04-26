import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils import data
user_emb_matrix=np.array(pd.read_csv('util/user_emb.csv',header=None))

''' 超参数'''
attr_num=18  #属性的个数
attr_present_dim=5
batch_size=128
hidden_dim=100 #G隐藏层维度
user_emb_dim=18
epochs=5
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
attr_emb=torch.randn(2*attr_num,attr_present_dim,requires_grad=True).float()
class TrainDataset(data.Dataset):
    def __init__(self):
        self.data=np.array(pd.read_csv("./train_data.csv"))
        self.emb=user_emb_matrix
    def __getitem__(self, index):
        train_data=self.data[index]
        userid=train_data[0]
        attr_id=np.int32(train_data[2][1:-1].split())
        attr_id=torch.tensor(torch.from_numpy(attr_id)).long()
        attr=torch.index_select(attr_emb,0,attr_id).reshape(-1,attr_num*attr_present_dim)
        real= self.emb[userid]
        real=torch.from_numpy(real).reshape(1,18).float()
        real=torch.cat((attr,real),1)
        return attr_id,attr,real
    def __len__(self):
        return len(self.data)
train=TrainDataset()
train_loader=data.DataLoader(train,shuffle=True,batch_size=batch_size)
#生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(attr_num*attr_present_dim,hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim,user_emb_dim),
            nn.LeakyReLU(0.2),
            nn.Tanh()
        )
    def forward(self,attr):
        return self.layer1(attr)
#判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(attr_num*attr_present_dim+user_emb_dim,hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim,user_emb_dim),
        )
    def forward(self,usr_emb):
        y=F.sigmoid(self.layer1(usr_emb))
        return y
g=Generator()

g.to(device)
d=Discriminator()
d.to(device)
critizen=nn.CrossEntropyLoss()
g_optimizer=optim.Adam(g.parameters(),lr=0.001)
d_optimizer=optim.Adam(d.parameters(),lr=0.001)
for epoch in range(epochs):
    for i,data in enumerate(train_loader):
        attr_id,attr,real_user=data
        attr_id.to(device),attr.to(device),real_user.to(device)
        real_lables=torch.ones(batch_size,1)
        real_lables.to(device)
        fake_lables=torch.zeros(batch_size,1)
        fake_lables.to(device)
        #训练判别器
        fake=g(attr)
        fake_emb=torch.cat((attr,fake),1)
        fake_usr=d(fake_emb)
        real_usr=d(real_user)
        real_loss=critizen(real_usr,real_lables)
        fake_loss=critizen(fake_usr,fake_lables)
        d_loss=real_loss+fake_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        #训练生成器
        fu=d(fake_usr)
        g_loss=critizen(fu,real_lables)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1)%200==0:
            print("epoch{},iter{},d_loss{:.4f},g_loss{:4f}".format(epoch,i,d_loss,g_loss))













