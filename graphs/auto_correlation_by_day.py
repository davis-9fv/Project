from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from Util import misc


def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


path = 'C:/tmp/bitcoin/'
#input_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'

dfx = pd.read_csv(path + input_file, header=0, sep=',')
print(dfx.shape)
max_lag=dfx.shape[0]
#max_lag = 1000
lags = [x for x in range(0, max_lag)]
correlations = [0 for x in range(0, max_lag)]

for i in range(0, max_lag-1):
    corr = dfx['Avg'].autocorr(lag=i)
    correlations[i] = corr * 100
    print('Corr: %.2f Lang: %i' % (corr, i))

misc.plot_one_line('Correlation by day', lags, correlations, 'Lag', 'Correlation')
