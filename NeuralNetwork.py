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

    cb = TDSCoinbaseData()
    start_date = '20200101'
    end_date = '20200531'

    df = cb.get_market_data(product, start_date, end_date, interval=60)
    data = df.to_numpy()
    relevant_data = torch.tensor(data[:, 1:6])
    print(relevant_data[0])
    print(relevant_data.shape)

    hist = np.zeros(epochs)

    for t in range(epochs):

        for i in range(data.shape[0]):
            output = model(relevant_data[i])
            loss = criterion(output, relevant_data[i+1])

        hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train_model(50)
