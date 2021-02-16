# import torch
# from torch import nn
# from torch.nn import functional as F
# from numpy import savetxt
# import csv
# import pandas as pd
# import io
# import numpy as np

# from torch import nn
# import torch.utils.model_zoo as model_zoo
# import torch.onnx

# class AlipesEnsembleNeuralNetwork(nn.Module):    
#     def __init__(self,
#                 input_size=250,
#                 hidden_size=96,
#                 num_networks=16,
#                 max_predict=1):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_networks = num_networks
#         self.max_predict = max_predict
#         self.W0 = nn.Parameter(torch.rand(num_networks, hidden_size, input_size) - 0.5)
#         self.prelu_z_slopes = nn.Parameter(torch.rand(num_networks) * 0.1)
#         self.prelu_r_slopes = nn.Parameter(torch.rand(num_networks) * 0.1)
        
#         self.Wz = nn.Parameter(torch.rand(num_networks, hidden_size) - 0.5)
#         self.Wr = nn.Parameter(torch.rand(num_networks, hidden_size) - 0.5)
#         self.z_scale = nn.Parameter(torch.rand(num_networks)) 
#         self.a = torch.tensor([[1,2,3], [4, 5, 6], [7,8,9]])
#         self.b = torch.tensor([[9,8,7], [6, 5, 4], [3,2,1]])
#         self.c = torch.tensor([9,8,7,6,5,4,3,2,1])
          
#     def _forward_first_layer(self, x):
#         batchsize = x.shape[0]
#         #first linear transformation
#         h = x.mm(self.W0.reshape(self.num_networks * self.hidden_size, self.input_size).t())
#         ha = self.b.mm(self.a.reshape(3,3).t())
#         h = h.reshape(batchsize, self.num_networks, self.hidden_size)
#         ha = ha.reshape(3,1,3)
#         hz = self.prelu_z_slopes[None, :, None] * h * (h < 0) + h * (h >= 0)  # parametric relu
#         hza = self.c[None, :, None] * ha * (ha < 0) + ha * (ha >= 0)  # parametric relu
#         hr = self.prelu_r_slopes[None, :, None] * h * (h < 0) + h * (h >= 0)  # parametric relu
#         return hz, hr    
            
#     def _forward_second_layers(self, hz, hr):
#         #dual second layer linear transformations
#         z = (hz * self.Wz[None]).sum(axis=-1)
#         r = (hr * self.Wr[None]).sum(axis=-1)
#         return z, r    
        
#     def _combine_z_and_r(self, z, r):
#         z = 2.0 * torch.sigmoid(self.z_scale[None] * z) - 1.0
#         r = F.softplus(r)
#         return r * z    

#     def _ensemble_predictions(self, y):
#         y = y.clamp(-self.max_predict, self.max_predict) # clip prediction of each network
#         y = y.mean(axis=-1) # average over networks
#         return y

#     def forward(self, x):
#         #savetxt('x_forward.csv', x, delimiter=',')
#         hz, hr = self._forward_first_layer(x)
#         z, r = self._forward_second_layers(hz, hr)
#         y = self._combine_z_and_r(z, r)
#         y = self._ensemble_predictions(y)
#         return y

# if __name__ == "__main__":
#     model = AlipesEnsembleNeuralNetwork()
#     batchsize = 1
#     x = torch.rand(batchsize, model.input_size)
    
#     # y = model(x)

#     #dynamic quantization
#     model_int8 = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
#     y = model_int8(x)

#     print(y)

    
#     # model.eval()
#     # torch.onnx.export(model, x,  "Alipesmodel.onnx", 
#     #     export_params=True, 
#     #     opset_version=10,
#     #     do_constant_folding=True,
#     #     input_names = ['input'],
#     #     output_names = ['output'],
#     #     dynamic_axes={'input' : {0 : 'batch_size'},
#     #     'output' : {0 : 'batch_size'}})


#     # print("Model's state_dict:")
#     # for param_tensor in model.state_dict():
#     #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#     # torch.save(model.state_dict(), "/Users/Amira/Desktop/test.zip")
#     # torch.save(model, "/Users/Amira/Desktop/test1.zip")

#     #print('Model output', y)

#     print('-'*100)
#     #for name, param in model.named_parameters():
#     for name, param in model_int8.named_parameters():
#         if param.requires_grad:
#             print('Name:', name)
#             npdata = param.data.numpy() # convert to ndarray
#             print('type', npdata.dtype)
#             print('Size:', npdata.shape)
#             print('Data:', npdata)
#             print('\n')
#     print('-'*100)