import pandas as pd
import torch

def read_csv_to_tensor(file_path, header=None):
    data = pd.read_csv(file_path, header=header)
    return torch.tensor(data.values, dtype=torch.float32)

def save_tensor_to_csv(tensor, file_path):
    import numpy as np
    np.savetxt(file_path, tensor.numpy(), delimiter=',')
