from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
from Util import algorithm
import numpy
from pandas import DataFrame


def compare_train(train_scaled, y_predicted):
    predictions = list()
    for i in range(len(train_scaled)):
        yhat = y_predicted[i]
        predictions.append(yhat)

    rmse = sqrt(mean_squared_error(raw_values[1:split + 1], predictions))
    return rmse


def compare_test(test_scaled, y_predicted):
    predictions = list()
    for i in range(len(test_scaled)):
        yhat = y_predicted[i]
        predictions.append(yhat)
    d = test_scaled[:, -1]
    rmse = sqrt(mean_squared_error(d, predictions))

    return rmse, predictions


print('main_passangers.py, raw data.')
print('LSTM and SGD cannot be calculated.')
series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')

date = series['Date'].values
date = numpy.delete(date, (-1), axis=0)

raw_values = series['Passangers'].values

size_raw_values = len(raw_values)
split = int(size_raw_values * 0.80)  # 0.80
print('raw_values:' + str(len(raw_values)))

print('Raw values total size: %i, Train size: %i, Test size: %i ' % (size_raw_values, split, size_raw_values - split))

supervised = data_misc.timeseries_to_supervised(raw_values, 1)

supervised_values = supervised.values
supervised_values = supervised_values[1:, :]

raw_values = numpy.delete(raw_values, (-1), axis=0)

print('Supervised:' + str(len(supervised_values)))
print('Raw values:' + str(len(raw_values)))

# split data into train and test-sets
train, test = supervised_values[0:split], supervised_values[split:]

# transform the scale of the data
# scaler, train_scaled, test_scaled = data_misc.scale(train, test)

x_train, y_train = train[:, 0:-1], train[:, -1]
x_test, y_test = test[:, 0:-1], test[:, -1]
x_test = [x_test[i] for i in range(len(x_test))]

print('------- Train -------')
# No Prediction
y_hat_predicted = y_train
rmse = compare_train(train, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_hat_predicted = x_train
rmse = compare_train(train, y_hat_predicted)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_hat_predicted = algorithm.elastic_net2(x_train, y_train, x_train)
rmse = compare_train(train, y_hat_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 5)
rmse = compare_train(train, y_hat_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 10)
rmse = compare_train(train, y_hat_predicted)
print('RMSE KNN(10)  %.3f' % (rmse))

# SGD
y_hat_predicted = algorithm.sgd_regressor(x_train, y_train, x_train)
rmse = compare_train(train, y_hat_predicted)
print('RMSE SGD     %.3f' % (rmse))

# LSTM
y_hat_predicted = algorithm.lstm(x_train, y_train, x_train, batch_size=1, nb_epoch=3, neurons=1)
rmse = compare_train(train, y_hat_predicted)
print('RMSE LSTM    %.3f' % (rmse))

print('------- Test --------')
# No Prediction
y_hat_predicted = y_test
rmse, y_predicted_real = compare_test(test, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_hat_predicted = x_test
rmse, y_predicted_real_dummy = compare_test(test, y_hat_predicted)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_hat_predicted = algorithm.elastic_net2(x_train, y_train, x_test)
rmse, y_predicted_en = compare_test(test, y_hat_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_test, 5)
rmse, y_predicted_knn5 = compare_test(test, y_hat_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_test, 10)
rmse, y_predicted_knn10 = compare_test(test, y_hat_predicted)
print('RMSE KNN(10) %.3f' % (rmse))

# SGD
y_hat_predicted = algorithm.sgd_regressor(x_train, y_train, x_test)
rmse, y_predicted_sgd = compare_test(test, y_hat_predicted)
print('RMSE SGD     %.3f' % (rmse))

# LSTM
y_hat_predicted = algorithm.lstm(x_train, y_train, x_test, batch_size=1, nb_epoch=3, neurons=1)
rmse, y_predicted_lstm = compare_test(test, y_hat_predicted)
print('RMSE LSTM   %.3f' % (rmse))

titles = ['X', 'Y', 'ElasticNet', 'KNN5', 'KNN10']
data = [x_test, y_test, y_predicted_en, y_predicted_knn5, y_predicted_knn10]
misc.plot_lines_graph('Raw Data, Test Data ', date[split:], titles, data)
misc.plot_data_graph2('Realvalue',date,raw_values)