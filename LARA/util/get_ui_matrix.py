import numpy as np
import pandas as pd

def get_ui_matrix():
    data=np.array(pd.read_csv('../data/train_data.csv'))
    ui=np.zeros(shape=(6040,2536),dtype=np.int32) #6040:用户数 ，2536：电影数目
    for i in data:
        ui[i[0]][i[2]]=1 #0列为用户id，2列为电影id，ui对应索引为一表示他们交互过
    save=pd.DataFrame(ui)
    save.to_csv('../util/train_ui_matrix.csv',index=0,header=None)


get_ui_matrix()