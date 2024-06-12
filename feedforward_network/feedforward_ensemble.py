import torch
from torch import nn
import pandas as pd

def read_csv_to_tensor(file_path, header=None):
    data = pd.read_csv(file_path, header=header)
    return torch.tensor(data.values, dtype=torch.float32)

class FeedforwardEnsembleNetwork(nn.Module):
    def __init__(self, input_size=256, hidden_size=96, num_networks=16, max_predict=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_networks = num_networks
        self.max_predict = max_predict
        self.W0 = nn.Parameter(read_csv_to_tensor('data/W0.csv'))
        self.prelu_z_slopes = nn.Parameter(read_csv_to_tensor('data/prelu_z_slopes.csv'))
        self.prelu_r_slopes = nn.Parameter(read_csv_to_tensor('data/prelu_r_slopes.csv'))
        self.Wz = nn.Parameter(read_csv_to_tensor('data/Wz.csv').reshape(num_networks, hidden_size))
        self.Wr = nn.Parameter(read_csv_to_tensor('data/Wr.csv').reshape(num_networks, hidden_size))
        self.z_scale = nn.Parameter(read_csv_to_tensor('data/z_scale.csv'))

    def forward(self, x):
        batchsize = x.shape[0]
        h = x.mm(self.W0.reshape(self.num_networks * self.hidden_size, self.input_size).t())
        h = h.reshape(batchsize, self.num_networks, self.hidden_size)
        
        # Ensure dimensions match for prelu_z_slopes and prelu_r_slopes
        prelu_z_slopes = self.prelu_z_slopes.view(1, self.num_networks, 1)
        prelu_r_slopes = self.prelu_r_slopes.view(1, self.num_networks, 1)

        hz = prelu_z_slopes * h * (h < 0) + h * (h >= 0)
        hr = prelu_r_slopes * h * (h < 0) + h * (h >= 0)

        z = (hz * self.Wz[None]).sum(axis=-1)
        r = (hr * self.Wr[None]).sum(axis=-1)
        
        y = r * (2.0 * torch.sigmoid(self.z_scale[None] * z) - 1.0)
        y = y.clamp(-self.max_predict, self.max_predict).mean(axis=-1)
        
        return y
