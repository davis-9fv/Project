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
shuffle_data = True
write_file = False
iterations = 1
x_iteration = [x for x in range(0, iterations)]
# y_rmse = [0 for x in range(0, iterations)]

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Iterations: %i' % (iterations))
print('Shuffle: %i' % (shuffle_data))

path = 'C:/tmp/bitcoin/'
#input_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_8_elasticnet_btc.csv'

columns = ['size','reward']
"""
columns = ['Open',
           'High', 'Low', 'Close', 'day_of_week', 'day_of_month', 'day_of_year', 'month_of_year',
           'year', 'week_of_year_column', 'transaction_count', 'input_count', 'output_count',
           'input_total', 'input_total_usd', 'output_total', 'output_total_usd', 'fee_total',
           'fee_total_usd', 'generation', 'reward', 'size', 'weight', 'stripped_size']
"""
window_size = 7 # 7
result = list()
y_rmse = [0 for x in range(0, len(columns))]
for i in range(0, len(columns)):
    print('')
    print('')
    print('Column: '+ columns[i])

    print('Window Size: %i' % (window_size))

    series = read_csv(path + input_file, header=0, sep=',')

    series = series.iloc[::-1]
    date = series['Date']
    weekday = series[columns[i]]
    avg = series['Avg']
    date = date.iloc[window_size:]
    date = date.values

    avg_values = avg.values
    weekday_raw_values = weekday.values
    avg_values = data_misc.timeseries_to_supervised(avg_values, window_size)

    # The first [Window size number] contains zeros which need to be cut.
    avg_values = avg_values.values[window_size:, :]
    # We cut from the beginning to pair with the supervised values.
    weekday_raw_values = weekday_raw_values[:-window_size]

    # Concatenate with numpy
    raw_values = np.concatenate((weekday_raw_values[:, None], avg_values), axis=1)
    if shuffle_data:
        raw_values = shuffle(raw_values)

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
    y_predicted_dummy = x_test[:, 0]
    rmse = compare(y_test, y_predicted_dummy)
    print('RMSE Dummy   %.3f' % (rmse))

    # ElasticNet
    y_predicted_en, y_future_en = algorithm.elastic_net(x_train, y_train, x_test, y_test)
    rmse = compare(y_test, y_predicted_en)
    print('RMSE Elastic %.3f' % (rmse))
    y_rmse[i] = ('%.3f') % rmse
    # print('Y_test')
    # print(y_test)
    print(result)

    titles = ['Y', 'ElasticNet']
    data = [y_test, y_predicted_en]

    date_test = date[split:]
    print('Length date test:' + str(len(date_test)))
    print('Length data test:' + str(len(y_test)))

print(columns)
print(y_rmse)

df = DataFrame({'Columns': columns,
                    'RMSE': y_rmse})
print(df)
if write_file:
    df.to_csv(path + output_file, header=True)

time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
