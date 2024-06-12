import os
import sys

# Get the absolute path of the directory that contains the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of the current directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from torch import optim, nn
from feedforward_network.feedforward_ensemble import FeedforwardEnsembleNetwork
from utils.data_utils import read_csv_to_tensor

def train():
    model = FeedforwardEnsembleNetwork()
    x = read_csv_to_tensor('data/x.csv').reshape(1000, model.input_size)
    target = read_csv_to_tensor('data/y_test.csv')
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs[:, None], target)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == "__main__":
    train()
