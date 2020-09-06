import torch


x = torch.tensor([[[1, 2, 3, 4], [2, 2, 3, 4]], [[1, 2, 3, 4], [2, 2, 3 ,4]]])
y = torch.tensor([[[5], [5]], [[5], [5]]])
print(x.shape)
x = x.view(2*2, -1)
y = y.view(2*2, -1)
print(x.shape)
print(y.shape)
o = torch.cat([x, y], dim=1)
print(o)
o=o.view(2, 2, -1)
print(o.shape)