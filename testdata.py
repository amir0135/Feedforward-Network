# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy
# from numpy import savetxt
# import csv


# input_size=3
# hidden_size=3
# num_networks=3
# max_predict=1

# W0 = torch.arange(num_networks* hidden_size* input_size)
# print("W0", W0.shape)
# W0 = torch.flatten(W0)
# savetxt('W0.csv', W0)

# prelu_z_slopes = torch.arange(num_networks)
# print("pre z", prelu_z_slopes.shape)
# prelu_z_slopes = torch.flatten(prelu_z_slopes)
# savetxt('prelu_z_slopes.csv', prelu_z_slopes)

# prelu_r_slopes = torch.arange(num_networks)
# print("pre r", prelu_r_slopes.shape)
# prelu_r_slopes = torch.flatten(prelu_r_slopes)
# savetxt('prelu_r_slopes.csv', prelu_r_slopes)

# Wz = torch.arange(num_networks* hidden_size)
# print("wz",Wz.shape)
# Wz = torch.flatten(Wz)  
# savetxt('Wz.csv', Wz)

# Wr = torch.arange(num_networks* hidden_size)
# print("wr", Wr.shape)
# Wr = torch.flatten(Wr)
# savetxt('Wr.csv', Wr)

# z_scale = torch.arange(num_networks)
# print("z_scale", z_scale.shape)
# z_scale = torch.flatten(z_scale)
# savetxt('z_scale.csv', z_scale)

# batchsize = 1
# x = torch.arange(batchsize* input_size)
# print("x", x.shape)
# x = torch.flatten(x)
# savetxt('x.csv', x)