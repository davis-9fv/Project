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

# This script predicts bitcoin price based on selected columns.

def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


time_start = datetime.datetime.now()
result = list()
shuffle_data = False
write_file = False
#use_columns = False
iterations = 1
x_iteration = [x for x in range(0, iterations)]

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Iterations: %i' % (iterations))
print('Shuffle: %i' % (shuffle_data))
#print('use_columns: %i' % (use_columns))

path = 'C:/tmp/bitcoin/'
#input_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_8_elasticnet_btc.csv'



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
columns = ['output_total_usd', 'input_total_usd',
           'Trend', 'year', 'fee_total_usd', 'generation', 'input_count', 'size'
    , 'fee_total', 'reward', 'output_count', 'weight']
print('12 best f_regression')

# 8 best f_regression
columns = ['output_total_usd', 'input_total_usd',
           'Trend', 'year', 'fee_total_usd', 'generation', 'input_count', 'size']
print('8 best f_regression')

"""

"""

# 12 best ExtraTreesClassifier
columns = [ 'day_of_year', 'day_of_month',
           'reward', 'generation', 'fee_total_usd', 'output_count',
           'output_total_usd', 'size','input_count', 'input_total_usd', 'output_total', 'transaction_count']
print('12 best ExtraTreesClassifier')




"""
# 8 best ExtraTreesClassifier
columns = ['day_of_year', 'day_of_month',
           'reward', 'generation', 'fee_total_usd', 'output_count',
           'output_total_usd', 'size','input_count']
print('8 best ExtraTreesClassifier')




"""
# 11 best f_regression - no high,low,close, open
columns = ['Trend', 'year', 'fee_total_usd', 'generation',
           'input_count', 'size', 'fee_total', 'reward', 'output_count',
           'weight', 'transaction_count']

print('11 best f_regression - no high,low,close, open')
"""


"""
# Union F_regression and ExtraTreesClassifier - no high,low,close, open
columns = ['output_total_usd','input_total_usd','fee_total_usd',
           'generation','input_count','size','reward','output_count'
    ,'day_of_year','day_of_month','output_total'
    ,'transaction_count','Trend','year','fee_total','weight']
print('Union F_regression and ExtraTreesClassifier - no high,low,close, open')


# Intersecion F_regression and ExtraTreesClassifier - no high,low,close, open
columns = ['output_total_usd','input_total_usd','fee_total_usd',
           'generation','input_count','size','reward','output_count']

print('Intersecion F_regression and ExtraTreesClassifier - no high,low,close, open')





# best 9 no usd - ExtraTreesClassifier - no high,low,close, open
columns = [ 'output_total','transaction_count','day_of_week'
    ,'input_total','generation', 'input_count', 'size', 'reward', 'output_count']
print('ExtraTreesClassifier - no high,low,close, open')

"""

"""
# best 9 no usd - F_regression - no high,low,close, open
columns = ['Trend', 'year', 'generation', 'input_count', 'size', 'fee_total', 'reward', 'output_count', 'weight']
print('F_regression - no high,low,close, open')


"""




window_size = 5  # 7
result = list()
y_rmse = [0 for x in range(0, len(columns))]

print('')
print('')
print('Columns: ' +str(columns))

print('Window Size: %i' % (window_size))

series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
series = series.iloc[::-1]

dfx = DataFrame()
for column in columns:
    dfx[column] = series[column]

date = series['Date']
avg = series['Avg']
date = date.iloc[window_size:]
date = date.values


avg_values = avg.values
avg_values = data_misc.timeseries_to_supervised(avg_values, window_size)
avg_values = avg_values.values[window_size:, :]
print('-----')
#raw_values = range(len(avg_values))
#raw_values = [i*0 for i in raw_values]

raw_values = avg_values

for column_name in columns:
    col = dfx[column_name]
    col_values = data_misc.timeseries_to_supervised(col, window_size)
    col_values = col_values.values[window_size:, :]

    # Concatenate with numpy
    raw_values = np.concatenate((col_values,raw_values), axis=1)




print(raw_values[0:3])


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
y_predicted_en, y_future_en = algorithm.elastic_net(x_train, y_train, x_test, y_test,normalize=True)
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
