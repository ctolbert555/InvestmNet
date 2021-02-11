import torch
import torch.nn as nn
import numpy as np
from TDSCoinbaseData import TDSCoinbaseData
import logging
logging.getLogger().setLevel(level=logging.ERROR)


class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()

        self.fc1 = nn.Linear(240, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 40)
        self.fc4 = nn.Linear(40, 60)

    def forward(self, x):

        out = x.reshape((4, 1, -1))
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out.reshape((4, 15, 4))


class FF2(nn.Module):
    def __init__(self):
        super(FF2, self).__init__()

        self.fc1 = nn.Linear(240, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, 15)

    def forward(self, x):

        out = x.reshape((1, 1, -1))
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out.reshape((1, 15, 1))


def MAPE(target, actual):
    return torch.mean(torch.abs((target-actual)/(torch.abs(target) + torch.abs(actual)/2)))


def MSE_DIFF(target, actual):
    targ_diff = target[:, 1:, :] - target[:, :-1, :]
    act_diff = actual[:, 1:, :] - actual[:, :-1, :]
    mse = nn.MSELoss()
    m_diff = mse(targ_diff, act_diff)
    return m_diff


def MAPE_DIFF(target, actual):
    return MAPE(target, actual) + MSE_DIFF(target, actual)


def train_model(load_path, save_path, epochs, product):
    model = FF2()
    if load_path != "":
        model.load_state_dict(torch.load(load_path))

    criterion = MAPE_DIFF
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    cb = TDSCoinbaseData()
    start_date = '20200101'
    end_date = '20200630'

    df = cb.get_market_data(product, start_date, end_date, interval=60)

    low = torch.tensor(df['low'].values).unsqueeze(0)
    hi = torch.tensor(df['high'].values).unsqueeze(0)
    o = torch.tensor(df['open'].values).unsqueeze(0)
    c = torch.tensor(df['close'].values).unsqueeze(0)

    tensor_data = torch.stack((low, hi, o, c), 2).float()
    print(tensor_data[:, 1:2, :].shape)
    print(tensor_data.shape)

    for t in range(epochs):
        avg_loss = torch.zeros(1)
        for i in range(tensor_data.shape[1] - 75):
            output = model(tensor_data[:, i:i+60, :])
            loss = criterion(output, tensor_data[:, i+60:i+75, 3:])
            if torch.isnan(loss):
                print("FAILED")
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("AVG LOSS")
        print(avg_loss / (tensor_data.shape[1] - 75))
        print("SAVING")
        torch.save(model.state_dict(), save_path)


def test_model(path, product):
    model = FF2()
    model.load_state_dict(torch.load(path))

    criterion = MAPE

    cb = TDSCoinbaseData()
    start_date = '20201001'
    end_date = '20201231'

    df = cb.get_market_data(product, start_date, end_date, interval=60)

    low = torch.tensor(df['low'].values).unsqueeze(0)
    hi = torch.tensor(df['high'].values).unsqueeze(0)
    o = torch.tensor(df['open'].values).unsqueeze(0)
    c = torch.tensor(df['close'].values).unsqueeze(0)

    tensor_data = torch.stack((low, hi, o, c), 2).float()

    avg_loss = torch.zeros(1)
    for i in range(tensor_data.shape[1] - 75):
        output = model(tensor_data[:, i:i + 60, :])
        print("DIFFERENCE")
        print(output, tensor_data[0, i + 60, 3])
        loss = criterion(output, tensor_data[:, i + 60:i + 75, 3:])
        avg_loss += loss
    avg_loss /= (tensor_data.shape[1] - 75)
    print("AVG LOSS: ")
    print(avg_loss)


products=['BTC-USD', 'ETH-BTC', 'LTC-BTC', 'BTC-EUR']

# train_model("usd_net.pth", "usd_md_net.pth", 5, products[0])
# train_model("eth_net.pth", "eth_md_net.pth", 5, products[1])
# train_model("ltc_net.pth", "ltc_md_net.pth", 5, products[2])
# train_model("eur_net.pth", "eur_md_net.pth", 5, products[3])
# test_model("usd_md_net.pth", products[0])
# test_model("eth_net.pth", products[1])
# test_model("ltc_mape_net.pth", products[2])
# test_model("eur_net.pth", products[3])
