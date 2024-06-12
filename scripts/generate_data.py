import os
import sys

# Get the absolute path of the directory that contains the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of the current directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch

from utils.data_utils import save_tensor_to_csv

def generate_data():
    input_size = 256
    hidden_size = 96
    num_networks = 16
    entry_size = 1000

    W0 = torch.rand(num_networks, hidden_size, input_size) - 0.5
    save_tensor_to_csv(W0.flatten(), 'data/W0.csv')
    
    prelu_z_slopes = torch.rand(num_networks) * 0.1
    save_tensor_to_csv(prelu_z_slopes, 'data/prelu_z_slopes.csv')

    prelu_r_slopes = torch.rand(num_networks) * 0.1
    save_tensor_to_csv(prelu_r_slopes, 'data/prelu_r_slopes.csv')

    Wz = torch.rand(num_networks, hidden_size) - 0.5
    save_tensor_to_csv(Wz, 'data/Wz.csv')

    Wr = torch.rand(num_networks, hidden_size) - 0.5
    save_tensor_to_csv(Wr, 'data/Wr.csv')

    z_scale = torch.rand(num_networks)
    save_tensor_to_csv(z_scale, 'data/z_scale.csv')

    x = torch.rand(entry_size, input_size)
    save_tensor_to_csv(x, 'data/x.csv')

    y = torch.sin(torch.sum(x, axis=1))
    save_tensor_to_csv(y, 'data/y_test.csv')

if __name__ == "__main__":
    generate_data()
