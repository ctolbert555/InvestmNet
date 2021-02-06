from TDSCoinbaseData import TDSCoinbaseData
from TDSTickGenerator import TDSTickGenerator
from TDSTransactionTracker import TDSTransactionTracker
import logging
logging.getLogger().setLevel(level=logging.ERROR)

cb = TDSCoinbaseData()

start_date = '20200101'
end_date = '20201231'
products = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'ETH-BTC', 'LTC-BTC']

for product in products:
    ## Only use overwrite=True when backfilling data. Otherwise, either specify overwrite=False or omit the arg entirely
    df = cb.get_market_data(product, start_date, end_date, interval=60, overwrite=True)

# Instantiate a tick generator
# tick_gen = TDSTickGenerator(cb, products, start_date, end_date, interval=60)
