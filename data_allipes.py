import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from numpy import savetxt
import csv

input_size=256
hidden_size=96
num_networks=16
max_predict=1
entry_size =1000

W0 = torch.rand(num_networks, hidden_size, input_size) - 0.5
print("W0", W0.shape)
W0 = torch.flatten(W0)
savetxt('W0.csv', W0)#, delimiter=',')
#W0 = W0.resize_((16, 95, 500))
#print(W0)

prelu_z_slopes = torch.rand(num_networks) * 0.1
print("pre z", prelu_z_slopes.shape)
prelu_z_slopes = torch.flatten(prelu_z_slopes)
savetxt('prelu_z_slopes.csv', prelu_z_slopes)

prelu_r_slopes = torch.rand(num_networks) * 0.1
print("pre r", prelu_r_slopes.shape)
prelu_r_slopes = torch.flatten(prelu_r_slopes)
savetxt('prelu_r_slopes.csv', prelu_r_slopes)

Wz = torch.rand(num_networks, hidden_size) - 0.5
print("wz",Wz.shape)
Wz = torch.flatten(Wz)  
savetxt('Wz.csv', Wz)

Wr = torch.rand(num_networks, hidden_size) - 0.5
print("wr", Wr.shape)
Wr = torch.flatten(Wr)
savetxt('Wr.csv', Wr)

z_scale = torch.rand(num_networks)
print("z_scale", z_scale.shape)
z_scale = torch.flatten(z_scale)
savetxt('z_scale.csv', z_scale)

batchsize = 1
x = torch.rand(entry_size, input_size)
print("x", x.shape)
xf = torch.flatten(x)
savetxt('x.csv', xf)


y = np.sin(np.sum(x.numpy(), axis = 1))
y = y.flatten()
savetxt('y_test.csv',y)