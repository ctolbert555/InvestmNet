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


def MAPE(target, actual):
    return torch.mean(torch.abs((target-actual)/((target+actual)/2)))


def train_model(load_path, save_path, epochs, products=['BTC-USD', 'ETH-BTC', 'LTC-BTC', 'BTC-EUR']):
    model = FF()
    if load_path != "":
        model.load_state_dict(torch.load(load_path))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    cb = TDSCoinbaseData()
    start_date = '20200101'
    end_date = '20200531'

    df_usd = cb.get_market_data(products[0], start_date, end_date, interval=60)
    df_eth = cb.get_market_data(products[1], start_date, end_date, interval=60)
    df_ltc = cb.get_market_data(products[2], start_date, end_date, interval=60)
    df_eur = cb.get_market_data(products[3], start_date, end_date, interval=60)

    usd_low = torch.tensor(df_usd['low'].values)
    eth_low = torch.tensor(df_eth['low'].values)
    ltc_low = torch.tensor(df_ltc['low'].values)
    eur_low = torch.tensor(df_eur['low'].values)
    tensor_low = torch.stack((usd_low, eth_low, ltc_low, eur_low), 0)
    usd_hi = torch.tensor(df_usd['high'].values)
    eth_hi = torch.tensor(df_eth['high'].values)
    ltc_hi = torch.tensor(df_ltc['high'].values)
    eur_hi = torch.tensor(df_eur['high'].values)
    tensor_high = torch.stack((usd_hi, eth_hi, ltc_hi, eur_hi), 0)
    usd_o = torch.tensor(df_usd['open'].values)
    eth_o = torch.tensor(df_eth['open'].values)
    ltc_o = torch.tensor(df_ltc['open'].values)
    eur_o = torch.tensor(df_eur['open'].values)
    tensor_open = torch.stack((usd_o, eth_o, ltc_o, eur_o), 0)
    usd_c = torch.tensor(df_usd['close'].values)
    eth_c = torch.tensor(df_eth['close'].values)
    ltc_c = torch.tensor(df_ltc['close'].values)
    eur_c = torch.tensor(df_eur['close'].values)
    tensor_close = torch.stack((usd_c, eth_c, ltc_c, eur_c), 0)
    tensor_data = torch.stack([tensor_low, tensor_high, tensor_open, tensor_close], 1).reshape((4, -1, 4)).float()
    print(tensor_data[:, 1:2, :].shape)
    print(tensor_data.shape)

    for t in range(epochs):
        avg_loss = torch.zeros(1)
        for i in range(tensor_data.shape[1] - 75):
            output = model(tensor_data[:, i:i+60, :])
            loss = criterion(output, tensor_data[:, i+60:i+75, :])
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("AVG LOSS")
        print(avg_loss / (tensor_data.shape[1] - 75))
        print("SAVING")
        torch.save(model.state_dict(), save_path)


def test_model(path, products=['BTC-USD', 'ETH-BTC', 'LTC-BTC', 'BTC-EUR']):
    model = FF()
    model.load_state_dict(torch.load(path))

    criterion = torch.nn.L1Loss()

    cb = TDSCoinbaseData()
    start_date = '20200601'
    end_date = '20200630'

    df_usd = cb.get_market_data(products[0], start_date, end_date, interval=60)
    df_eth = cb.get_market_data(products[1], start_date, end_date, interval=60)
    df_ltc = cb.get_market_data(products[2], start_date, end_date, interval=60)
    df_eur = cb.get_market_data(products[3], start_date, end_date, interval=60)

    usd_low = torch.tensor(df_usd['low'].values)
    eth_low = torch.tensor(df_eth['low'].values)
    ltc_low = torch.tensor(df_ltc['low'].values)
    eur_low = torch.tensor(df_eur['low'].values)
    tensor_low = torch.stack((usd_low, eth_low, ltc_low, eur_low), 0)
    usd_hi = torch.tensor(df_usd['high'].values)
    eth_hi = torch.tensor(df_eth['high'].values)
    ltc_hi = torch.tensor(df_ltc['high'].values)
    eur_hi = torch.tensor(df_eur['high'].values)
    tensor_high = torch.stack((usd_hi, eth_hi, ltc_hi, eur_hi), 0)
    usd_o = torch.tensor(df_usd['open'].values)
    eth_o = torch.tensor(df_eth['open'].values)
    ltc_o = torch.tensor(df_ltc['open'].values)
    eur_o = torch.tensor(df_eur['open'].values)
    tensor_open = torch.stack((usd_o, eth_o, ltc_o, eur_o), 0)
    usd_c = torch.tensor(df_usd['close'].values)
    eth_c = torch.tensor(df_eth['close'].values)
    ltc_c = torch.tensor(df_ltc['close'].values)
    eur_c = torch.tensor(df_eur['close'].values)
    tensor_close = torch.stack((usd_c, eth_c, ltc_c, eur_c), 0)
    tensor_data = torch.stack([tensor_low, tensor_high, tensor_open, tensor_close], 1).reshape((4, -1, 4)).float()

    avg_loss = torch.zeros(1)
    for i in range(tensor_data.shape[1] - 75):
        output = model(tensor_data[:, i:i + 60, :])
        # print(output[:, :, 3], tensor_data[:, i + 60:i + 61, 3])
        loss = criterion(output[:, :, 3], tensor_data[:, i + 60:i + 75, 3])
        avg_loss += loss
    avg_loss /= (tensor_data.shape[1] - 75)
    print("AVG LOSS: ")
    print(avg_loss)


#train_model("", "multi_net.pth", 50)
#test_model("multi_net.pth")
