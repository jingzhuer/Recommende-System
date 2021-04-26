import torch
a=torch.ones(5,5)
b=torch.tensor([0,1,3,4,5])
a[0]=b
print(a[0])