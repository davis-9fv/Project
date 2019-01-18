from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import data_misc
from sklearn.utils import shuffle
import datetime
from Util import algorithm
import numpy as np
from itertools import combinations


# This script executes all the possible cobinations between columns in order to predict
# bitcoin price.

def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))


def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


time_start = datetime.datetime.now()
result_columns = list()
result_rmse = list()
shuffle_data = False
write_file = True
iterations = 1
x_iteration = [x for x in range(0, iterations)]

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Iterations: %i' % (iterations))
print('Shuffle: %i' % (shuffle_data))

path = 'C:/tmp/bitcoin/'
# input_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_8_elasticnet_btc_combination_ft_result.csv'
"""
columns = ['reward', 'week_of_year_column', 'Trend', 'transaction_count', 'input_total', 'output_total', 'day_of_month',
           'day_of_year']
"""
columns = ['Open',
           'High', 'Low', 'Close', 'day_of_week', 'day_of_month', 'day_of_year', 'month_of_year',
           'year', 'week_of_year_column', 'transaction_count', 'input_count', 'output_count',
           'input_total', 'input_total_usd', 'output_total', 'output_total_usd', 'fee_total',
           'fee_total_usd', 'generation', 'reward', 'size', 'weight', 'stripped_size']

# Driver Function
for r in range(1, len(columns) + 1):
    column_combination_array = rSubset(columns, r)
    column_combination_array = np.asarray(column_combination_array)

    for column_combination in column_combination_array:
        window_size = 7  # 7
        result = list()
        y_rmse = [0 for x in range(0, len(column_combination))]

        print('')
        print('')
        print('Columns: ' + str(column_combination))

        print('Window Size: %i' % (window_size))

        series = read_csv(path + input_file, header=0, sep=',')
        series = series.iloc[::-1]

        dfx = DataFrame()
        for column in column_combination:
            dfx[column] = series[column]

        date = series['Date']
        avg = series['Avg']
        date = date.iloc[window_size:]
        date = date.values

        avg_values = avg.values
        raw_values = dfx.values
        avg_values = data_misc.timeseries_to_supervised(avg_values, window_size)
        # print(raw_values)
        # print(avg_values)

        # The first [Window size number] contains zeros which need to be cut.
        avg_values = avg_values.values[window_size:, :]
        # We cut from the beginning to pair with the supervised values.
        raw_values = raw_values[:-window_size, :]

        # print(raw_values)
        # print(avg_values)
        print('-----')
        # Concatenate with numpy
        raw_values = np.concatenate((raw_values, avg_values), axis=1)
        # print(raw_values)
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
        y_predicted_en, y_future_en = algorithm.elastic_net(x_train, y_train, x_test, y_test, normalize=True)
        rmse = compare(y_test, y_predicted_en)
        print('RMSE Elastic %.3f' % (rmse))
        result_columns.append(column_combination)
        result_rmse.append(rmse)

        titles = ['Y', 'ElasticNet']
        data = [y_test, y_predicted_en]

        date_test = date[split:]
        print('Length date test:' + str(len(date_test)))
        print('Length data test:' + str(len(y_test)))

    time_end = datetime.datetime.now()
    print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
    print('Duration of the script: %s' % (str(time_end - time_start)))

print('Results:')
result_columns = np.asarray(result_columns)
result_rmse = np.asarray(result_rmse)

for i in range(0, len(result_columns)):
    value = result_rmse[i]
    print('%.3f' % value + "  " + str(result_columns[i]))

if write_file:
    df = DataFrame({'Columns': result_columns,
                    'RMSE': result_rmse})
    df.to_csv(path + output_file, header=True)

print('')
print('End')
