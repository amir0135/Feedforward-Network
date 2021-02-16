#!/usr/bin/env python3
import torch
from torch import nn
from torch.nn import functional as F
from numpy import savetxt
import csv
import pandas as pd
import io
import numpy as np

import torch.optim as optim
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


# def readcsv(file):
#     data = pd.read_csv(file, header=None)
#     data = np.array(data)
#     data = np.reshape(data,(data.size))
#     data = torch.Tensor(data)
#     return data

def readcsv(file):
    data = pd.read_csv(file)
    data = torch.tensor(data.values)
    data = torch.reshape(data, (data.size))
    data = np.argwhere(data)
    return data
#print(data.shape)

class AlipesEnsembleNeuralNetwork(nn.Module):    

    def __init__(self,
            input_size=256,
            hidden_size=96,
            num_networks=16,
            max_predict=1,
            W0='W0.csv',
            prelu_z='prelu_z_slopes.csv',
            prelu_r='prelu_r_slopes.csv',
            Wz='Wz.csv',
            Wr='Wr.csv',
            z_scale='z_scale.csv'
          ):

            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_networks = num_networks
            self.max_predict = max_predict
            self.W0 = readcsv(W0)
            self.prelu_z_slopes = readcsv(prelu_z)
            self.prelu_r_slopes = readcsv(prelu_r)
            self.Wz = readcsv(Wz).reshape(num_networks, hidden_size)
            self.Wr = readcsv(Wr).reshape(num_networks, hidden_size)
            self.z_scale = readcsv(z_scale)




    def _forward_first_layer(self, x):
        batchsize = x.shape[0]
        h = x.mm(self.W0.reshape(self.num_networks * self.hidden_size, self.input_size).t())
        h = h.reshape(batchsize, self.num_networks, self.hidden_size)
        hz = self.prelu_z_slopes[None, :, None] * h * (h < 0) + h * (h >= 0)  # parametric relu
        hr = self.prelu_r_slopes[None, :, None] * h * (h < 0) + h * (h >= 0)  # parametric relu
        return hz, hr    
            
    def _forward_second_layers(self, hz, hr):
        #dual second layer linear transformations
        z = (hz * self.Wz[None]).sum(axis=-1)
        r = (hr * self.Wr[None]).sum(axis=-1)
        return z, r    
        
    def _combine_z_and_r(self, z, r):
        z = 2.0 * torch.sigmoid(self.z_scale[None] * z) - 1.0
        r = F.softplus(r)
        return r * z    

    def _ensemble_predictions(self, y):
        y = y.clamp(-self.max_predict, self.max_predict) # clip prediction of each network
        y = y.mean(axis=-1) # average over networks
        return y

    def forward(self, x):
        #savetxt('x_forward.csv', x, delimiter=',')
        hz, hr = self._forward_first_layer(x)
        z, r = self._forward_second_layers(hz, hr)
        y = self._combine_z_and_r(z, r)
        y = self._ensemble_predictions(y)
        return y

if __name__ == "__main__":
    model = AlipesEnsembleNeuralNetwork()
    batchsize = 1
    #x = readcsv('X_data.csv').reshape(1, batchsize, model.input_size)
    x = torch.tensor(pd.read_csv('X_data.csv').values)[:,None]
    print(x.shape)
    target = torch.tensor(pd.read_csv('y.csv').values)
    print(target.shape)
    y = model(x)

#define cross-entropy loss
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# #strain model by looping over our data iterator, and feed the inputs to the modelwork and optimize.
# for epoch in range(10):  # loop over the dataset multiple times

#     # zero the parameter gradients
#     optimizer.zero_grad()

#     # forward + backward + optimize
#     outputs = model(x)
#     loss = criterion(outputs, target)
#     loss.backward()
#     optimizer.step()

# print('Finished Training')


    
    # model.eval()
    # torch.onnx.export(model, x,  "Alipesmodel.onnx", 
    #     export_params=True, 
    #     opset_version=10,
    #     do_constant_folding=True,
    #     input_names = ['input'],
    #     output_names = ['output'],
    #     dynamic_axes={'input' : {0 : 'batch_size'},
    #     'output' : {0 : 'batch_size'}})


    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # torch.save(model.state_dict(), "/Users/Amira/Desktop/test.zip")
    # torch.save(model, "/Users/Amira/Desktop/test1.zip")

    # print('Model output', y)

    # print('-'*100)
    # #print('Model parameters:', list(model_int8.named_parameters()))
    # for name, param in model_int8.named_parameters():
    #     print(param)
    #     if param.requires_grad:
    #         print('Name:', name)
    #         npdata = param.data.numpy() # convert to ndarray
    #         print('type', npdata.dtype)
    #         print('Size:', npdata.shape)
    #         #print('Data:', npdata)
    #         print('\n')
    # print('-'*100)
