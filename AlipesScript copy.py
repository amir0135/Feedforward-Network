#!/usr/bin/env python3
import torch
from torch import nn
from torch.nn import functional as F
from numpy import savetxt
import csv
import pandas as pd
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from torch import optim 

def readcsv(file):
    data = pd.read_csv(file, header=None)
    data = np.array(data)
    data = np.reshape(data,(data.size))
    data = torch.Tensor(data)
    return data

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
            z_scale='z_scale.csv',
            X_real = 'X_data.csv'

          ):

            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_networks = num_networks
            self.max_predict = max_predict
            self.W0 = nn.Parameter(readcsv(W0))
            self.prelu_z_slopes = nn.Parameter(readcsv(prelu_z))
            self.prelu_r_slopes = nn.Parameter(readcsv(prelu_r))
            self.Wz = nn.Parameter(readcsv(Wz).reshape(num_networks, hidden_size))
            self.Wr = nn.Parameter(readcsv(Wr).reshape(num_networks, hidden_size))
            self.z_scale = nn.Parameter(readcsv(z_scale))
            #self.X_real = nn.Parameter(readcsv(X_real))
            #print(X_real.shape)




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
    #x = nn.Parameter(readcsv('x.csv').reshape(batchsize, model.input_size))
    x = readcsv('x.csv').reshape(1000, model.input_size)
    x = nn.Parameter(x)

    #x = (batchsize, model.input_size)
    
    y = model(x)

    #dynamic quantization
    model_int8 = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    y1 = model_int8(x)

    print(y, 'y', y1, 'y1')

  
##define cross-entropy loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
target = readcsv('y_test.csv')

#strain model by looping over our data iterator, and feed the inputs to the modelwork and optimize.
for epoch in range(10):  # loop over the dataset multiple times
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = model(x)
    #outputs = outputs.reshape(1, 1000)
    loss = criterion([outputs[0]], [target[0]])
    loss.backward()
    optimizer.step()

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

    print('-'*100)
    #print('Model parameters:', list(model_int8.named_parameters()))
    for name, param in model_int8.named_parameters():
        print(param)
        if param.requires_grad:
            print('Name:', name)
            npdata = param.data.numpy() # convert to ndarray
            print('type', npdata.dtype)
            print('Size:', npdata.shape)
            #print('Data:', npdata)
            print('\n')
    print('-'*100)