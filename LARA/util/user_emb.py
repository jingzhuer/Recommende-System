import pandas as pd
import numpy as np
#得到用户的编码
def user_emb():
    data=pd.read_csv('../data/train_data.csv',usecols=['userId','gerne'])
    data['tem']=data['gerne'].str.split('[',expand=True)[1]
    data['tem1']=data['tem'].str.split(']',expand=True)[0]
    user=np.array(data['userId'])
    attr=np.array(data['tem1'])
    user_present=np.zeros(shape=(6040,18),dtype=np.int32)  #6040：用户数，18：电影的属性个数，电影的种类数
    for i in range(len(user)):
        attr_list=np.int32(attr[i].split(','))
        for j in attr_list:
            user_present[user[i]][j]+=1.0 #只要交互过一次某类型的电影，就在该属性值上加1
    save=pd.DataFrame(user_present)
    save['Col_sum']=save.apply(lambda x:x.sum(),axis=1)  #6040*19
    save=np.array(save,dtype=np.float32)
    for i in range(6040):
        tt=save[i][-1]
        if tt!=0.0:
            for j in range(18):
                save[i][j]=save[i][j]/tt
    save=pd.DataFrame(save)
    save.to_csv('../util/user_emb.csv',index=0,header=None,columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

user_emb()










