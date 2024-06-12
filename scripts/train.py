import torch
from torch import optim, nn
from models.feedforward_ensemble import FeedforwardEnsembleNetwork
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
