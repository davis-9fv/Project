from sklearn.utils import shuffle
import datetime
from Util import algorithm
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc


def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


time_start = datetime.datetime.now()
result = list()
shuffle_data = True
write_file = False

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Shuffle: %i' % (shuffle_data))

path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin_usd_11_10_2018.csv'

window_size = 5  # best is 5
result = list()
print('')
print('')
print('Window Size: %i' % (window_size))

series = read_csv(path + input_file, header=0, sep=',')

series = series.iloc[::-1]
date = series['Date']
avg = series['Avg']
date = date.iloc[window_size:]
date = date.values

avg_values = avg.values

supervised = data_misc.timeseries_to_supervised(avg_values, window_size)
# The first [Window size number] contains zeros which need to be cut.
supervised = supervised.values[window_size:, :]

if shuffle_data:
    supervised = shuffle(supervised)

size_supervised = len(supervised)
split = int(size_supervised * 0.80)

train, test = supervised[0:split], supervised[split:]


x_train, y_train = train[:, 0:-1], train[:, -1]
x_test, y_test = test[:, 0:-1], test[:, -1]

print('Size size_supervised %i' % (size_supervised))

print('------- Test --------')
# No Prediction
y_hat_predicted_es = y_test
rmse = compare(y_test, y_hat_predicted_es)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_predicted_dummy_es = x_test[:, 0]
rmse = compare(y_test, y_predicted_dummy_es)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_predicted_en_es, y_future_en_es = algorithm.elastic_net(x_train, y_train, x_test, y_test, normalize=True)
rmse = compare(y_test, y_predicted_en_es)
print('RMSE Elastic %.3f' % (rmse))

# y_future_en = compare(y_test, y_future_en_es)



# SGD
y_predicted_sgd_es = algorithm.sgd_regressor(x_train, y_train, x_test)
rmse = compare(y_test, y_predicted_sgd_es)
print('RMSE SGD     %.3f' % (rmse))

# Lasso
y_predicted_la_sc = algorithm.lasso(x_train, y_train, x_test, normalize=True)
rmse = compare(y_test, y_predicted_la_sc)
print('RMSE Lasso   %.3f' % (rmse))




time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
