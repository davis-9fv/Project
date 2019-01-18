from pandas import read_csv
from pandas import concat
from pandas import RangeIndex
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import data_misc
from sklearn.utils import shuffle
import datetime
from Util import algorithm
import numpy as np


# This script creates a window size, we can select several features and we also create
# another column which represents the price of bitcoin of the previous year (also created
# with a window size).
# The data is based on daily data.
def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


time_start = datetime.datetime.now()
result = list()
shuffle_data = False
write_file = False
use_columns = True
iterations = 1
x_iteration = [x for x in range(0, iterations)]

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Iterations: %i' % (iterations))
print('Shuffle: %i' % (shuffle_data))
print('use_columns: %i' % (use_columns))

path = 'C:/tmp/bitcoin/'
# input_file = 'bitcoin_usd_11_10_2018.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_8_elasticnet_btc_combination_ft_result_1year.csv'

"""
columns = ['transaction_count','input_total','output_total','Trend']

columns = ['Open',
           'High', 'Low', 'Close', 'day_of_week', 'day_of_month', 'day_of_year', 'month_of_year',
           'year', 'week_of_year_column', 'transaction_count', 'input_count', 'output_count',
           'input_total', 'input_total_usd', 'output_total', 'output_total_usd', 'fee_total',
           'fee_total_usd', 'generation', 'reward', 'size', 'weight', 'stripped_size', 'Trend']

"""

"""

# 12 best f_regression
columns = ['High', 'Low', 'Close', 'Open', 'output_total_usd', 'input_total_usd',
           'Trend', 'year', 'fee_total_usd', 'generation', 'input_count', 'size']
print('12 best f_regression')

# 8 best f_regression
columns = ['High', 'Low', 'Close', 'Open', 'output_total_usd', 'input_total_usd',
           'Trend', 'year']
print('8 best f_regression')

"""


"""

# 12 best ExtraTreesClassifier
columns = ['Low', 'High', 'Close', 'Open', 'day_of_year', 'day_of_month',
           'reward', 'generation', 'fee_total_usd', 'output_count',
           'output_total_usd', 'size']
print('12 best ExtraTreesClassifier')


# 8 best ExtraTreesClassifier
columns = ['Low', 'High', 'Close', 'Open', 'day_of_year', 'day_of_month',
           'reward', 'generation']
print('8 best ExtraTreesClassifier')



# 11 best f_regression - no high,low,close, open
columns = ['Trend', 'year', 'fee_total_usd', 'generation',
           'input_count', 'size', 'fee_total', 'reward', 'output_count',
           'weight', 'transaction_count','week_of_year_column']

print('11 best f_regression - no high,low,close, open')
"""

# Intersecion F_regression and ExtraTreesClassifier - no high,low,close, open
columns = ['output_total_usd','input_total_usd','fee_total_usd',
           'generation','input_count','size','reward','output_count']

print('Intersecion F_regression and ExtraTreesClassifier - no high,low,close, open')
"""

# Union F_regression and ExtraTreesClassifier - no high,low,close, open
columns = ['output_total_usd','input_total_usd','fee_total_usd',
           'generation','input_count','size','reward','output_count'
    ,'day_of_year','day_of_month','output_total'
    ,'transaction_count','Trend','year','fee_total','weight']

print('Union F_regression and ExtraTreesClassifier - no high,low,close, open')
"""


window_size = 5  # 7
result = list()
y_rmse = [0 for x in range(0, len(columns))]

print('')
print('')
print('Columns: ' + str(columns))

print('Window Size: %i' % (window_size))

series = read_csv(path + input_file, header=0, sep=',')
series = series.iloc[::-1]
# The index is ordered in ascending mode
series.index = RangeIndex(len(series.index))

# We drop index which contains 29/02/
index_to_drop = series.loc[series['Date'] == '2016-02-29'].index.values.astype(int)[0]
series = series.drop(series.index[index_to_drop])

# The index is ordered again due to the a index was dropped
series.index = RangeIndex(len(series.index))

# We create a new dataframe so we can work with it
dfx = DataFrame()
for column in columns:
    dfx[column] = series[column]

date = series['Date']
avg = series['Avg']

lag = 365
avg, avg_previous = data_misc.slide_data(avg.values, lag)
date, date_previous = data_misc.slide_data(series['Date'].values, lag)

df = DataFrame({'date_previous': date_previous,
                'avg_previous': avg_previous,
                'date': date,
                'avg': avg})

print(df.head(10))

# The data is made supervised
avg = data_misc.timeseries_to_supervised(avg, window_size)
avg_previous = data_misc.timeseries_to_supervised(avg_previous, window_size)

# The first [Window size number] contains zeros which need to be cut.
avg = avg.values[window_size:, :]
avg_previous = avg_previous.values[window_size:, :]

# series we no longer use series obj because contains columns that we don't need
raw_values = dfx.values[lag:]
# print(series)

# We cut the values which are zero
raw_values = raw_values[:-window_size, :]
date_previous = DataFrame(date_previous)
date_previous = date_previous.values[:-window_size]

# We concatenate the two created columns with the data. Para usar esto, descomentar los dos
# raw_values = np.concatenate((raw_values, avg_previous), axis=1)
# raw_values = np.concatenate((raw_values, avg), axis=1)

# Este comentar cuando no se use las columnas
if use_columns:
    # raw_values_tmp = np.concatenate((date_previous, avg_previous), axis=1)
    # raw_values = np.concatenate((raw_values_tmp, raw_values), axis=1)
    raw_values = np.concatenate((avg_previous, raw_values), axis=1)
    raw_values = np.concatenate((raw_values, avg), axis=1)
else:
    raw_values = np.concatenate((avg_previous, avg), axis=1)

# print(raw_values[0:10])

if shuffle_data:
    raw_values = shuffle(raw_values, random_state=9)
    # raw_values = shuffle(raw_values)

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

# Lasso
y_predicted_en = algorithm.lasso(x_train, y_train, x_test, normalize=True)
rmse = compare(y_test, y_predicted_en)
print('RMSE Lasso %.3f' % (rmse))

titles = ['Y', 'ElasticNet']
data = [y_test, y_predicted_en]

date_test = date[split:]
print('Length date test:' + str(len(date_test)))
print('Length data test:' + str(len(y_test)))

print(columns)

time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
