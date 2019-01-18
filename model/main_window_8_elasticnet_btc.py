from pandas import read_csv
from pandas import concat
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import data_misc
from sklearn.utils import shuffle
import datetime
from Util import algorithm
import numpy as np


def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


time_start = datetime.datetime.now()
result = list()
shuffle_data = False
write_file = False
iterations = 1
x_iteration = [x for x in range(0, iterations)]
# y_rmse = [0 for x in range(0, iterations)]

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Iterations: %i' % (iterations))
print('Shuffle: %i' % (shuffle_data))

path = 'C:/tmp/bitcoin/'
# input_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_8_elasticnet_btc.csv'

window_size = 5  # 7
result = list()
print('')
print('')

print('Window Size: %i' % (window_size))

# To pair with the other models, this model gets 1438 first rows.
series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
# print(series.tail(5))
series = series.iloc[::-1]
date = series['Date']
avg = series['Avg']
date = date.iloc[window_size:]
date = date.values

avg_values = avg.values
avg_values = data_misc.timeseries_to_supervised(avg_values, window_size)

# The first [Window size number] contains zeros which need to be cut.
avg_values = avg_values.values[window_size:, :]

raw_values = avg_values
if shuffle_data:
    raw_values = shuffle(raw_values, random_state=9)

# print(raw_values)

size_raw_values = len(raw_values)
split = int(size_raw_values * 0.80)

train, test = raw_values[0:split], raw_values[split:]
x_train, y_train = train[:, 0:-1], train[:, -1]
x_test, y_test = test[:, 0:-1], test[:, -1]

print('Size raw_values %i' % (size_raw_values))

print('------- Test --------')
# No Prediction
y_hat_predicted = y_test
rmse = compare(y_test, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_predicted_dummy = x_test[:, -1]
rmse = compare(y_test, y_predicted_dummy)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_predicted_en, y_future_en = algorithm.elastic_net(x_train, y_train, x_test, y_test, normalize=True)
rmse = compare(y_test, y_predicted_en)
print('RMSE Elastic %.3f' % (rmse))

titles = ['Y', 'ElasticNet']
data = [y_test, y_predicted_en]

date_test = date[split:]
print('Length date test:' + str(len(date_test)))
print('Length data test:' + str(len(y_test)))


time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
