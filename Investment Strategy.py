from TDSCoinbaseData import TDSCoinbaseData
from TDSTickGenerator import TDSTickGenerator
from TDSTransactionTracker import TDSTransactionTracker
import logging
import numpy
logging.getLogger().setLevel(level=logging.ERROR)

cb = TDSCoinbaseData()

start_date = '20200101'
end_date = '20201231'
products = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'ETH-BTC', 'LTC-BTC']
isHolding = False
takers_fee = 0.0018

for product in products:
    ## Only use overwrite=True when backfilling data. Otherwise, either specify overwrite=False or omit the arg entirely
    df = cb.get_market_data(product, start_date, end_date, interval=60, overwrite=False)

# Instantiate a tick generator
tick_gen = TDSTickGenerator(cb, products, start_date, end_date, interval=60)
trans_tracker = TDSTransactionTracker(start_date, end_date, holdings={'BTC' : 1.0})



#TEST findHigh
down_slope = [188, 187, 186, 185, 184, 183]
up_slope = [188, 189, 190, 191, 194, 195]
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
print("Troph: ", findHigh(troph))

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

# # Helper funciton to check if a trade should be profitable
# def isProfitable(isHolding, current_tick, last_price):
#     if not isHolding:
#         # When selling, and the last price is not greater than curent price plus the fee, return false
#         if not last_price > (current_tick + takers_fee):
#             return False
#     else:
#         # When buying, and the last price is not greater than curent price plus the fee, return false
#         #TODO Fix this
#         if not (last_price < (current_tick + takers_fee)):
#             return False
#     return True



# Function that checks predictions to see if this ticker is a buy/sell

def makeTrade(isHolding, current_tick, tickers):
    ticker_price = tick.p.btc_usd.close
    #TODO if looking to buy, True, otherwise false
    if not isHolding:
        # Scenerio where current_tick is a trough
        #TODO correct for numpy array
        if(findLow(tickers) >= ticker_price):
            trans_tracker.make_trade(current_tick, 'BTC-USD', 'buy', -1)
            isHolding = False
    else:
        # Scenerio where current_tick is peak
        if findHigh(tickers) <= ticker_price:
            trans_tracker.make_trade(current_tick, 'BTC-USD', 'sell', -1)
            isHolding = True
       # Scenerio where in future there will be a better buy
        else if


tick = tick_gen.get_tick()
last_price = tick.tick.p.btc_usd.close


while tick is not None:
    tick = tick_gen.get_tick()
    if tick is not None:
        #TODO get predicted tickers from NeuralNetwork
        makeTrade(isHolding, tick, down_slope)
