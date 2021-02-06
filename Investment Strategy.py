from TDSCoinbaseData import TDSCoinbaseData
from TDSTickGenerator import TDSTickGenerator
from TDSTransactionTracker import TDSTransactionTracker
import logging
import numpy as np
logging.getLogger().setLevel(level=logging.ERROR)

cb = TDSCoinbaseData()

start_date = '20201001'
end_date = '20201231'
products = ['BTC-USD', 'ETH-BTC', 'LTC-BTC']
isHolding = True
takers_fee = 0.0018

print("Data grabbed")
# Instantiate a tick generator
tick_gen = TDSTickGenerator(cb, products, start_date, end_date, interval=60)
trans_tracker = TDSTransactionTracker(start_date, end_date, holdings={'BTC': 1.0})
df_usd = cb.get_market_data(products[0], start_date, end_date, interval=60, overwrite=False)
df_eth = cb.get_market_data(products[1], start_date, end_date, interval=60, overwrite=False)
df_ltc = cb.get_market_data(products[2], start_date, end_date, interval=60, overwrite=False)

print("Generator and Tracker created")

#TEST findHigh
down_slope = [188, 187, 186, 185, 184, 183]
up_slope = [188, 189, 190, 191, 194, 50000]
peak = [188, 189, 190, 191, 194, 195, 194, 193, 192]
troph = [188, 187, 186, 185, 184, 183, 188, 189, 190, 191, 194, 195]
# Helper function to find the highest ticker in a sequence
def findHigh(tickers):
    #TODO reconfigure for correct data storage type
    highest_ticker = tickers[0]
    for ticker in tickers:
        if ticker > highest_ticker:
            highest_ticker = ticker
    return highest_ticker

print("TEST findHigh: ")
print("Down Slope: ", findHigh(down_slope))
print("Up Slope: ", findHigh(up_slope))
print("Peak: ", findHigh(peak))
print("Trough: ", findHigh(troph))

# Helper funciton to find the lowest ticker in a sequence
def findLow(tickers):
    #TODO reconfigure for correct data storage type
    lowest_ticker = tickers[0]
    for ticker in tickers:
        if ticker < lowest_ticker:
            lowest_ticker = ticker
    return lowest_ticker

print("TEST findLow: ")
print("Down Slope: ", findLow(down_slope))
print("Up Slope: ", findLow(up_slope))
print("Peak: ", findLow(peak))
print("Troph: ", findLow(troph))


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


# Function to check if we are currently holding bitcoin
def checkIfHolding():
    if trans_tracker.get_holdings()["BTC"] == 0.0:
        return False
    return True


def checkIfUSD():
    if ("USD" not in trans_tracker.get_holdings()) or trans_tracker.get_holdings()["USD"] == 0.0:
        return False
    return True

def checkIfETH():
    if ("ETH" not in trans_tracker.get_holdings()) or trans_tracker.get_holdings()["ETH"] == 0.0:
        return False
    return True

def checkIfLTC():
    if ("LTC" not in trans_tracker.get_holdings()) or trans_tracker.get_holdings()["LTC"] == 0.0:
        return False
    return True


# Function that checks predictions to see if this ticker is a buy/sell


def makeTrade(current_tick, tickers, last_price):
    usd_val = 1/tick.p.btc_usd.close
    eth_val = tick.p.eth_btc.close
    ltc_val = tick.p.ltc_btc.close
    #TODO if looking to buy, True, otherwise false
    if checkIfHolding():
        # Scenario where current_tick is a trough
        #TODO correct for numpy array
        if tickers[0].min() >= usd_val:
            # BUY USD
            if tick.p.btc_usd.volume * 0.5 > trans_tracker.get_holdings()["BTC"]:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'sell', -1)
            else:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'sell', tick.p.btc_usd.volume * 0.4999999)
        if tickers[1].min() >= eth_val:
            # BUY ETH
            if tick.p.eth_btc.volume * 0.5 > trans_tracker.get_holdings()["BTC"] / eth_val:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'buy', -1)
            else:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'buy', (tick.p.eth_btc.volume * 0.499999) * eth_val)
        if tickers[2].min() >= ltc_val:
            # BUY LTC
            if tick.p.ltc_btc.volume * 0.5 > trans_tracker.get_holdings()["BTC"] / ltc_val:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'buy', -1)
            else:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'buy', tick.p.ltc_btc.volume * 0.499999 * ltc_val)
    if checkIfUSD():
        # Scenario where current_tick is peak
        if tickers[0].max() <= usd_val:
            if tick.p.btc_usd.volume * 0.5 > trans_tracker.get_holdings()["USD"] * usd_val:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'buy', -1)
            else:
                trans_tracker.make_trade(current_tick, 'BTC-USD', 'buy', tick.p.btc_usd.volume * 0.499999 / usd_val)
    if checkIfETH():
        # Scenario where current_tick is peak
        if tickers[1].max() <= eth_val:
            if tick.p.eth_btc.volume * 0.5 > trans_tracker.get_holdings()["ETH"]:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'sell', -1)
            else:
                trans_tracker.make_trade(current_tick, 'ETH-BTC', 'sell', tick.p.eth_btc.volume * 0.4999999)
    if checkIfLTC():
        # Scenario where current_tick is peak
        if tickers[2].max() <= ltc_val:
            if tick.p.ltc_btc.volume * 0.5 > trans_tracker.get_holdings()["LTC"]:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'sell', -1)
            else:
                trans_tracker.make_trade(current_tick, 'LTC-BTC', 'sell', tick.p.ltc_btc.volume * 0.4999999)


def model_predictions(frame):
    step = 25
    return np.stack((1/df_usd['close'].values[frame:frame+step],
                    df_eth['close'].values[frame:frame+step],
                    df_ltc['close'].values[frame:frame+step]), 0)


tick = tick_gen.get_tick()
last_price = tick.p.btc_usd.close

frame = 0
while tick is not None:
    tick = tick_gen.get_tick()
    frame += 1
    if tick is not None:
        #TODO get predicted tickers from NeuralNetwork
        makeTrade(tick, model_predictions(frame), last_price)
trans_tracker.plot_btc_holdings()
print(trans_tracker.get_holdings())
print(trans_tracker.get_sharpe_ratio())
trans_tracker.dump_trades('example_trades.json')
