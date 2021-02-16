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

train = pd.read_csv('X_data.csv')
train_tensor = torch.tensor(train.values)
t = np.argwhere(train_tensor)
print(t.shape)