from numpy.lib.function_base import append
from torch.functional import Tensor
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
class PMF:
    def __init__(self,usernum,movienum,k,lamdau=0.01,lamdav=0.01):
        self.P=torch.randn(usernum,k,dtype=float,requires_grad=True)
        self.q=torch.randn(k,movienum,dtype=float,requires_grad=True)
        self.lamdau=lamdau
        self.lamdav=lamdav
        self.I=torch.ones(usernum,movienum)
        self.optimizer=optim.Adam([{'params':self.P},{'params':self.q}],lr=0.01)
        self.critizen=nn.MSELoss()

    
    def read_data(self): 
        user_ids_dict, rated_item_ids_dict = {},{}
        u_idx, i_idx = 0,0
        data=[]
        f=open(r'recommended\PMF\ml-100k\u.data')
        for line in f.readlines():
            u,i,r,_=line.split('\t')
        if u not in user_ids_dict:
            user_ids_dict[u]=u_idx
            u_idx+=1
        if i not in rated_item_ids_dict:
            rated_item_ids_dict[i]=i_idx
            i_idx+=1
        data.append([user_ids_dict[u],rated_item_ids_dict[i],r])
        f.close()
        array=np.array(data)
        R=np.zeros([943,1682],dtype=float)
        row,col,rate=array[:,0].astype(int),array[:,1].astype(int),array[:,2].astype(float)
        R[row,col]=rate
        return R   
    def predict(self):
        R_pred=torch.mm(self.P,self.q)
        R_pred=F.sigmoid(R_pred)
        return R_pred
    
    def train(self,p,q):
        R=self.read_data()
        R=torch.from_numpy(R)
        R=F.sigmoid(R)
        iters=[]
        losses=[]
        for iter in range(200):
            R_pred=self.predict()
            loss=0.5*self.critizen(R,R_pred)+self.lamdau*(torch.square(self.q).sum())+self.lamdav*(torch.square(self.P).sum())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("iter:{},loss:{:.4f}".format(iter,loss))
            iters.append(iter)
            losses.append(loss)
        plt.plot(iters,losses)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()
model=PMF(943,1682,10)
model.train(model.P,model.q)
print(model.predict())






    


            
    


    

        

    

   
