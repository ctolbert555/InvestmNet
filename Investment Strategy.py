from TDSCoinbaseData import TDSCoinbaseData
from TDSTickGenerator import TDSTickGenerator
from TDSTransactionTracker import TDSTransactionTracker
import logging
import torch
import numpy as np
import NeuralNetwork
logging.getLogger().setLevel(level=logging.ERROR)

cb = TDSCoinbaseData()
starts = ['20201001', '20200701', '20190601']
ends = ['20201231', '20200930', '20190831']

#Global variables
index = 0
start_date = starts[index]
end_date = ends[index]
products = ['BTC-USD', 'ETH-BTC', 'LTC-BTC', 'BTC-EUR']
isHolding = True
takers_fee = 0.0018
# Initialize stack
currency_stack = torch.zeros((4, 60, 5))
projection = torch.zeros((4, 1, 4))

print("Data grabbed")
# Instantiate a tick generator
tick_gen = TDSTickGenerator(cb, products, start_date, end_date, interval=60)
trans_tracker = TDSTransactionTracker(start_date, end_date, holdings={'BTC': 1.0})
df_usd = cb.get_market_data(products[0], start_date, end_date, interval=60, overwrite=False)
df_eth = cb.get_market_data(products[1], start_date, end_date, interval=60, overwrite=False)
df_ltc = cb.get_market_data(products[2], start_date, end_date, interval=60, overwrite=False)
df_eur = cb.get_market_data(products[3], start_date, end_date, interval=60, overwrite=False)
print("Generator and Tracker created")

def updateStack(tick):
    '''
    Updates the stack with the current tick
    :return: None
    '''
    #shift stack
    currency_stack[:, 1:60, :] = currency_stack[:, 0:59, :]

    '''USD'''
    currency_type = 0
    # Update high
    currency_stack[currency_type, 0, 0] = tick.p.btc_usd.high
    # Update low
    currency_stack[currency_type, 0, 1] = tick.p.btc_usd.low
    # Update open
    currency_stack[currency_type, 0, 2] = tick.p.btc_usd.open
    # Update close
    currency_stack[currency_type, 0, 3] = tick.p.btc_usd.close
    # Update volume
    currency_stack[currency_type, 0, 4] = tick.p.btc_usd.volume

    '''ETH'''
    currency_type = 1
    # Update high
    currency_stack[currency_type, 0, 0] = tick.p.eth_btc.high
    # Update low
    currency_stack[currency_type, 0, 1] = tick.p.eth_btc.low
    # Update open
    currency_stack[currency_type, 0, 2] = tick.p.eth_btc.open
    # Update close
    currency_stack[currency_type, 0, 3] = tick.p.eth_btc.close
    # Update volume
    currency_stack[currency_type, 0, 4] = tick.p.eth_btc.volume

    '''LTC'''
    currency_type = 2
    # Update high
    currency_stack[currency_type, 0, 0] = tick.p.ltc_btc.high
    # Update low
    currency_stack[currency_type, 0, 1] = tick.p.ltc_btc.low
    # Update open
    currency_stack[currency_type, 0, 2] = tick.p.ltc_btc.open
    # Update close
    currency_stack[currency_type, 0, 3] = tick.p.ltc_btc.close
    # Update volume
    currency_stack[currency_type, 0, 4] = tick.p.ltc_btc.volume

    '''EUR'''
    currency_type = 3
    # Update high
    currency_stack[currency_type, 0, 0] = tick.p.btc_eur.high
    # Update low
    currency_stack[currency_type, 0, 1] = tick.p.btc_eur.low
    # Update open
    currency_stack[currency_type, 0, 2] = tick.p.btc_eur.open
    # Update close
    currency_stack[currency_type, 0, 3] = tick.p.btc_eur.close
    # Update volume
    currency_stack[currency_type, 0, 4] = tick.p.btc_eur.volume

#------
#------
# Helper funciton to check if a trade should be profitable
def isProfitable(isHolding, current_price, last_price):
    if isHolding:
        # When selling, and the last price is not greater than curent price plus the fee, return false
        if last_price <= (current_price * (1 + takers_fee)):
            return False
    else:
        # When buying, and the last price is not greater than curent price plus the fee, return false
        if last_price >= (current_price * (1 - takers_fee)):
            return False
    return True
#------
#------
# Function to check if we are currently holding BTC
def checkIfHolding():
    if trans_tracker.get_holdings()["BTC"] == 0.0:
        return False
    return True

# Function to check if we are currently holding USD
def checkIfUSD():
    if ("USD" not in trans_tracker.get_holdings()) or trans_tracker.get_holdings()["USD"] == 0.0:
        return False
    return True

# Function to check if we are currently holding EUR
def checkIfEUR():
    if ("EUR" not in trans_tracker.get_holdings()) or trans_tracker.get_holdings()["EUR"] == 0.0:
        return False
    return True

# Function to check if we are currently holding ETH
def checkIfETH():
    if ("ETH" not in trans_tracker.get_holdings()) or trans_tracker.get_holdings()["ETH"] == 0.0:
        return False
    return True

# Function to check if we are currently holding LTC
def checkIfLTC():
    if ("LTC" not in trans_tracker.get_holdings()) or trans_tracker.get_holdings()["LTC"] == 0.0:
        return False
    return True
#------
#------
# Function that checks predictions to see if this ticker is a buy/sell
def makeTrade(current_tick, tickers):
    usd_val = 1/tick.p.btc_usd.close
    eth_val = tick.p.eth_btc.close
    ltc_val = tick.p.ltc_btc.close
    eur_val = 1/tick.p.btc_eur.close

    fraction = 1
    #TODO if looking to buy, True, otherwise false
    if checkIfHolding():
        # Scenario where current_tick is a trough
        #TODO correct for numpy array
        if tickers[0].min() >= usd_val:
            # BUY USD
            print("BUY USD")
            if tick.p.btc_usd.volume * 0.5 > trans_tracker.get_holdings()["BTC"] / fraction:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'sell', trans_tracker.get_holdings()["BTC"] / fraction)
            else:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'sell', tick.p.btc_usd.volume * 0.4999999)
        if tickers[1].min() >= eth_val:
            # BUY ETH
            print("BUY ETH")
            if tick.p.eth_btc.volume * 0.5 > (trans_tracker.get_holdings()["BTC"] / fraction) / eth_val:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'buy', trans_tracker.get_holdings()["BTC"] / fraction)
            else:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'buy', (tick.p.eth_btc.volume * 0.499999) * eth_val)
        if tickers[2].min() >= ltc_val:
            # BUY LTC
            print("BUY LTC")
            if tick.p.ltc_btc.volume * 0.5 > (trans_tracker.get_holdings()["BTC"] / fraction) / ltc_val:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'buy', trans_tracker.get_holdings()["BTC"] / fraction)
            else:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'buy', tick.p.ltc_btc.volume * 0.499999 * ltc_val)
        if tickers[3].min() >= eur_val:
            # BUY EUR
            print("BUY EUR")
            if tick.p.btc_eur.volume * 0.5 > trans_tracker.get_holdings()["BTC"] / fraction:
                trans_tracker.make_trade(current_tick, 'BTC-EUR', 'sell', trans_tracker.get_holdings()["BTC"] / fraction)
            else:
                trans_tracker.make_trade(current_tick, 'BTC-EUR', 'sell', tick.p.btc_eur.volume * 0.4999999)
    if checkIfUSD():
        # Scenario where current_tick is peak
        if tickers[0].max() <= usd_val:
            print("SELL USD")
            if tick.p.btc_usd.volume * 0.5 > trans_tracker.get_holdings()["USD"] * usd_val:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'buy', trans_tracker.get_holdings()["USD"])
            else:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'buy', tick.p.btc_usd.volume * 0.499999 / usd_val)
    if checkIfETH():
        # Scenario where current_tick is peak
        if tickers[1].max() <= eth_val:
            print("SELL ETH")
            if tick.p.eth_btc.volume * 0.5 > trans_tracker.get_holdings()["ETH"]:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'sell', trans_tracker.get_holdings()["ETH"])
            else:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'sell', tick.p.eth_btc.volume * 0.4999999)
    if checkIfLTC():
        # Scenario where current_tick is peak
        if tickers[2].max() <= ltc_val:
            print("SELL LTC")
            if tick.p.ltc_btc.volume * 0.5 > trans_tracker.get_holdings()["LTC"]:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'sell', trans_tracker.get_holdings()["LTC"])
            else:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'sell', tick.p.ltc_btc.volume * 0.4999999)
    if checkIfEUR():
        # Scenario where current_tick is peak
        if tickers[3].max() <= eur_val:
            print("SELL EUR")
            if tick.p.btc_eur.volume * 0.5 > trans_tracker.get_holdings()["EUR"] * eur_val:
                trans_tracker.make_trade(current_tick, 'BTC-EUR', 'buy', trans_tracker.get_holdings()["EUR"])
            else:
                trans_tracker.make_trade(current_tick, 'BTC-EUR', 'buy', tick.p.btc_eur.volume * 0.499999 / eur_val)
#------
#------
def model_predictions(frame, step=15):
    # Volatility threshold
    # threshold = 12
    # # print(np.average(df_usd['volume'].values[frame:frame+step]))
    # #TODO convert dataframe to tensor code
    # if np.average(df_usd['volume'].values[frame - step :frame]) > 3 * threshold:
    #     step = 15
    # elif np.average(df_usd['volume'].values[frame - step:frame]) > 2 * threshold:
    #     step = 20
    # elif np.average(df_usd['volume'].values[frame - step:frame]) > threshold:
    #     step = 30

    return np.stack((1/df_usd['close'].values[frame:frame+step],
                    df_eth['close'].values[frame:frame+step],
                    df_ltc['close'].values[frame:frame+step],
                    1/df_eur['close'].values[frame:frame+step]), 0)
#------


def nn_predictions(usd, eth, ltc, eur):
    current_stack = currency_stack[0:1, :, 0:4].clone().detach()
    usd_out = usd(current_stack)
    current_stack = currency_stack[1:2, :, 0:4].clone().detach()
    eth_out = eth(current_stack)
    current_stack = currency_stack[2:3, :, 0:4].clone().detach()
    ltc_out = ltc(current_stack)
    current_stack = currency_stack[3:4, :, 0:4].clone().detach()
    eur_out = eur(current_stack)
    return torch.stack((1/usd_out, eth_out, ltc_out, 1/eur_out), 0)


tick = tick_gen.get_tick()
# last_price = tick.p.btc_usd.close
usd_net = NeuralNetwork.FF2()
usd_net.load_state_dict(torch.load("usd_net.pth"))
eth_net = NeuralNetwork.FF2()
eth_net.load_state_dict(torch.load("eth_net.pth"))
ltc_net = NeuralNetwork.FF2()
ltc_net.load_state_dict(torch.load("ltc_mape_net.pth"))
eur_net = NeuralNetwork.FF2()
eur_net.load_state_dict(torch.load("eur_net.pth"))
# Which frame we are currently on
frame = 0
# How many minutes we should wait before trading (usually because we need enough data)
wait = 60
while tick is not None:
    updateStack(tick)
    frame += 1
    #TODO Save tickers into tensor code so we can perform volatility calculations
    if frame > wait:
        #TODO get predicted tickers from NeuralNetwork
        predictions = nn_predictions(usd_net, eth_net, ltc_net, eur_net)
        # predictions = model_predictions(frame)
        # print(predictions[:, :, 0])
        makeTrade(tick, predictions)
    tick = tick_gen.get_tick()


# Display findings
trans_tracker.plot_btc_holdings()
print(trans_tracker.get_holdings())
print(trans_tracker.get_sharpe_ratio())
trans_tracker.dump_trades('usd_trades_'+str(index)+'.json')
