import torch
import torch.nn as nn
import numpy as np
from TDSCoinbaseData import TDSCoinbaseData


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.hidden_dim = 160
        self.num_layers = 3

        self.gru = nn.GRU(5, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def train_model(epochs, product='BTC-USD'):
    model = GRU()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch_size = 16

    cb = TDSCoinbaseData()
    start_date = '20200101'
    end_date = '20200531'

    df = cb.get_market_data(product, start_date, end_date, interval=60)
    tensor_low = torch.tensor(df['low'].values)
    tensor_high = torch.tensor(df['high'].values)
    tensor_open = torch.tensor(df['open'].values)
    tensor_close = torch.tensor(df['close'].values)
    tensor_volume = torch.tensor(df['volume'].values)
    tensor_data = torch.stack([tensor_low, tensor_high, tensor_open, tensor_close, tensor_volume], 1)
    print(tensor_data[0])
    print(tensor_data.shape)

    for t in range(epochs):

        batch = 0
        for i in range(tensor_data.shape[0]-1):
            print('.')
            output = model(tensor_data[i])
            loss = criterion(output, tensor_data[i+1])
            batch += 1
            if batch >= batch_size:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch = 0


train_model(50)
