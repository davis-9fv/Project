from sklearn.utils import shuffle
import datetime
from Util import algorithm
from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
import numpy as np

def compare(y_test, y_predicted):
    predictions = list()
    for i in range(len(y_test)):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)

        #Stationary
        d = avg_values[split + window_size - 1:]
        yhat = data_misc.inverse_difference(d, yhat, len(y_test) + 1 - i)

        predictions.append(yhat)

    d = avg_values[split + window_size + 1:]
    #d = avg_values[split + window_size :]
    rmse = sqrt(mean_squared_error(d, predictions))
    #rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions

seed = 5
np.random.seed(seed)
time_start = datetime.datetime.now()
result = list()
shuffle_data = False
write_file = False
use_columns = True

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Shuffle: %i' % (shuffle_data))
print('use_columns: %i' % (use_columns))

path = 'C:/tmp/bitcoin/'
#input_file = 'bitcoin_usd_11_10_2018.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'



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


# 8 best ExtraTreesClassifier
columns = ['day_of_year', 'day_of_month',
           'reward', 'generation', 'fee_total_usd', 'output_count',
           'output_total_usd', 'size','input_count']
print('8 best ExtraTreesClassifier')
"""




"""
# 11 best f_regression - no high,low,close, open
columns = ['Trend', 'year', 'fee_total_usd', 'generation',
           'input_count', 'size', 'fee_total', 'reward', 'output_count',
           'weight', 'transaction_count']

print('11 best f_regression - no high,low,close, open')
"""

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
# best 9 no usd - ExtraTreesClassifier - no high,low,close, open
columns = [ 'output_total','transaction_count','day_of_week'
    ,'input_total','generation', 'input_count', 'size', 'reward', 'output_count']
print('ExtraTreesClassifier - no high,low,close, open')


# best 9 no usd - F_regression - no high,low,close, open
columns = ['Trend', 'year', 'generation', 'input_count', 'size', 'fee_total', 'reward', 'output_count', 'weight']
print('F_regression - no high,low,close, open')

"""




window_size = 5 # 7
result = list()
print('')
print('')
print('Window Size: %i' % (window_size))


# To pair with the other models, this model gets 1438 first rows.
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
raw_values = dfx.values
# Stationary Data
diff_values = data_misc.difference(avg_values, 1)
#diff_values= avg_values

supervised = data_misc.timeseries_to_supervised(diff_values, window_size)
# The first [Window size number] contains zeros which need to be cut.
supervised = supervised.values[window_size:, :]

# We cut the first or last element to pair with the diff values.
#raw_values = raw_values[1:, :]
raw_values = raw_values[:-1, :]

# We cut from the beginning to pair with the supervised values.
raw_values = raw_values[:-window_size, :]

print('-----')
# Concatenate with numpy
if use_columns:
    supervised = np.concatenate((raw_values, supervised), axis=1)




if shuffle_data:
    supervised = shuffle(supervised, random_state=9)

size_supervised = len(supervised)
split = int(size_supervised * 0.80)

train, test = supervised[0:split], supervised[split:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

print('Size size_supervised %i' % (size_supervised))

print('------- Test --------')
# No Prediction
y_hat_predicted_es = y_test
rmse, y_hat_predicted = compare(y_test, y_hat_predicted_es)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_predicted_dummy_es = x_test[:, -1]
rmse, y_predicted_dummy = compare(y_test, y_predicted_dummy_es)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_predicted_en_es, y_future_en_es = algorithm.elastic_net(x_train, y_train, x_test, y_test, normalize=False)
rmse, y_predicted_en = compare(y_test, y_predicted_en_es)
print('RMSE Elastic %.3f' % (rmse))

# y_future_en = compare(y_test, y_future_en_es)

# KNN5
y_predicted_knn5_es = algorithm.knn_regressor(x_train, y_train, x_test, 5)
rmse, y_predicted_knn5 = compare(y_test, y_predicted_knn5_es)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_predicted_knn10_es = algorithm.knn_regressor(x_train, y_train, x_test, 10)
rmse, y_predicted_knn10 = compare(y_test, y_predicted_knn10_es)
print('RMSE KNN(10) %.3f' % (rmse))

# SGD
y_predicted_sgd_es = algorithm.sgd_regressor(x_train, y_train, x_test)
rmse, y_predicted_sgd = compare(y_test, y_predicted_sgd_es)
print('RMSE SGD     %.3f' % (rmse))

# Lasso
y_predicted_la_sc = algorithm.lasso(x_train, y_train, x_test, normalize=False)
rmse, y_predicted_la = compare(y_test, y_predicted_la_sc)
print('RMSE Lasso   %.3f' % (rmse))

# LSTM
y_predicted_lstm = algorithm.lstm(x_train, y_train, x_test, batch_size=1, nb_epoch=60, neurons=14)
rmse, y_predicted_lstm = compare(y_test, y_predicted_lstm)
print('RMSE LSTM    %.3f' % (rmse))

titles = ['Y', 'ElasticNet', 'KNN5', 'KNN10', 'SGD', 'Lasso', 'LSTM']
data = [y_hat_predicted, y_predicted_en, y_predicted_knn5, y_predicted_knn10, y_predicted_sgd, y_predicted_la,
        y_predicted_lstm]
# titles = ['Y', 'ElasticNet', 'ElasticNet Future', 'KNN5', 'KNN10', 'SGD']
# data = [y_test, y_predicted_en, y_future_en, y_predicted_knn5, y_predicted_knn10]
# y_future_en = y_future_en[1]
# data = [y_hat_predicted, y_predicted_en, y_future_en, y_predicted_knn5, y_predicted_knn10, y_predicted_sgd]
date_test = date[split + 1:]
print('Length date test:' + str(len(date_test)))
print('Length data test:' + str(len(y_test)))
misc.plot_lines_graph('Stationary - Normalization,Test Data ', date_test, titles, data)

time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
