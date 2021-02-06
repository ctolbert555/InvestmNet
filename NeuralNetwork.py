import torch
import torch.nn as nn
import numpy as np
from TDSCoinbaseData import TDSCoinbaseData


class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()

        self.fc1 = nn.Linear(240, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 40)
        self.fc4 = nn.Linear(40, 4)

    def forward(self, x):

        out = x.reshape((1, 1, -1))
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


def train_model(load_path, save_path, epochs, product='BTC-USD'):
    model = FF()
    if load_path != "":
        model.load_state_dict(torch.load(load_path))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    batch_size = 64

    cb = TDSCoinbaseData()
    start_date = '20200101'
    end_date = '20200531'

    df = cb.get_market_data(product, start_date, end_date, interval=60)
    tensor_low = torch.tensor(df['low'].values)
    tensor_high = torch.tensor(df['high'].values)
    tensor_open = torch.tensor(df['open'].values)
    tensor_close = torch.tensor(df['close'].values)
    tensor_data = torch.stack([tensor_low, tensor_high, tensor_open, tensor_close], 1).reshape((1, -1, 4)).float()
    print(tensor_data[:, 1:2, :].shape)
    print(tensor_data.shape)

    for t in range(epochs):
        avg_loss = torch.zeros(1)
        for i in range(tensor_data.shape[1] - 60):
            output = model(tensor_data[:, i:i+60, :])
            loss = criterion(output, tensor_data[:, i+60:i+61, :])
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("AVG LOSS")
        print(avg_loss / (tensor_data.shape[1] - 60))
        print("SAVING")
        torch.save(model.state_dict(), save_path)


def test_model(path, product='BTC-USD'):
    model = FF()
    model.load_state_dict(torch.load(path))

    criterion = torch.nn.L1Loss()

    cb = TDSCoinbaseData()
    start_date = '20200601'
    end_date = '20200630'

    df = cb.get_market_data(product, start_date, end_date, interval=60)
    tensor_low = torch.tensor(df['low'].values)
    tensor_high = torch.tensor(df['high'].values)
    tensor_open = torch.tensor(df['open'].values)
    tensor_close = torch.tensor(df['close'].values)
    tensor_data = torch.stack([tensor_low, tensor_high, tensor_open, tensor_close], 1).reshape(
        (1, -1, 4)).float()

    avg_loss = torch.zeros(1)
    for i in range(tensor_data.shape[1] - 60):
        output = model(tensor_data[:, i:i + 60, :])
        print(output[:, :, 3], tensor_data[:, i + 60:i + 61, 3])
        loss = criterion(output[:, :, 3], tensor_data[:, i + 60:i + 61, 3])
        avg_loss += loss
    avg_loss /= (tensor_data.shape[1] - 60)
    print("AVG LOSS: ")
    print(avg_loss)


train_model("btc_usd.pth", "btc_usd2.pth", 50)
#test_model("btc_usd2.pth")
