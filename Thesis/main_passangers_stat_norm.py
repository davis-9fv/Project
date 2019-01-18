from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
from Util import algorithm
import numpy


def compare_train(train_scaled, y_predicted):
    predictions = list()
    for i in range(len(train_scaled)):
        X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        # Se agrega split+1 ya para que sea consecuente con la data para realizar el diff.
        yhat = data_misc.inverse_difference(raw_values[:split + 1], yhat, len(train_scaled) + 1 - i)
        # print(yhat)
        predictions.append(yhat)
    d = raw_values[1:split + 1]
    # Se empieza desde uno ya que el primer dato no se puede tomar en cuenta por diff.
    rmse = sqrt(mean_squared_error(d, predictions))
    return rmse


def compare_test(test_scaled, y_predicted):
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = y_predicted[i]
        # predictions.append(yhat)
        yhat = data_misc.invert_scale(scaler, X, yhat)
        yhat = data_misc.inverse_difference(y_raw_supervised[split:], yhat, len(test_scaled) + 1 - i)
        # print(yhat)
        predictions.append(yhat)

    # Se aumenta uno ya que el primer dato no se puede tomar en cuenta por diff.
    d = raw_values[1 + split:]
    # d = test_scaled[:, -1]
    rmse = sqrt(mean_squared_error(d, predictions))

    return rmse, predictions


print('main_passangers.py, stationary and normalization.')
series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')

# transform data to be stationary
date = series['Date'].values
date = numpy.delete(date, (0), axis=0)

raw_values = series['Passangers'].values

size_raw_values = len(raw_values)
split = int(size_raw_values * 0.80)  # 0.80
print('raw_values:' + str(len(raw_values)))

raw_supervised = data_misc.timeseries_to_supervised(raw_values, 1)
raw_supervised = raw_supervised.values
y_raw_supervised = raw_supervised[:, -1]
# y_raw_supervised = numpy.delete(y_raw_supervised, (0), axis=0)
print('y_raw_supervised:' + str(len(y_raw_supervised)))

print('raw_supervised:' + str(len(raw_supervised)))

print('Raw values total size: %i, Train size: %i, Test size: %i ' % (size_raw_values, split, size_raw_values - split))

diff_values = data_misc.difference(raw_values, 1)
print('Diff:' + str(len(diff_values)))

# raw_values = numpy.delete(raw_values, (0), axis=0)
print('raw_values deleted first:' + str(len(raw_values)))

supervised = data_misc.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
print('Supervised:' + str(len(supervised_values)))

# split data into train and test-sets
train, test = supervised_values[0:split], supervised_values[split:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
x_test = [x_test[i] for i in range(len(x_test))]

print('------- Train -------')
# No Prediction
y_hat_predicted = y_train
rmse = compare_train(train_scaled, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_hat_predicted = x_train
rmse = compare_train(train_scaled, y_hat_predicted)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_hat_predicted = algorithm.elastic_net2(x_train, y_train, x_train, normalize=False)
rmse = compare_train(train_scaled, y_hat_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 5)
rmse = compare_train(train_scaled, y_hat_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 10)
rmse = compare_train(train_scaled, y_hat_predicted)
print('RMSE KNN(10)  %.3f' % (rmse))

# SGD
y_hat_predicted = algorithm.sgd_regressor(x_train, y_train, x_train)
rmse = compare_train(train_scaled, y_hat_predicted)
print('RMSE SGD     %.3f' % (rmse))

# LSTM
y_hat_predicted = algorithm.lstm(x_train, y_train, x_train, batch_size=1, nb_epoch=3, neurons=1)
rmse = compare_train(train_scaled, y_hat_predicted)
print('RMSE LSTM    %.3f' % (rmse))

print('------- Test --------')
# No Prediction
y_hat_predicted = y_test
rmse, y_predicted_real = compare_test(test_scaled, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_hat_predicted = x_test
rmse, y_predicted_real_dummy = compare_test(test_scaled, y_hat_predicted)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_hat_predicted = algorithm.elastic_net2(x_train, y_train, x_test, normalize=False)
rmse, y_predicted_en = compare_test(test_scaled, y_hat_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_test, 5)
rmse, y_predicted_knn5 = compare_test(test_scaled, y_hat_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_test, 10)
rmse, y_predicted_knn10 = compare_test(test_scaled, y_hat_predicted)
print('RMSE KNN(10) %.3f' % (rmse))

# SGD
y_hat_predicted = algorithm.sgd_regressor(x_train, y_train, x_test)
rmse, y_predicted_sgd = compare_test(test_scaled, y_hat_predicted)
print('RMSE SGD     %.3f' % (rmse))

# Lasso
y_hat_predicted = algorithm.lasso(x_train, y_train, x_test, normalize=False)
rmse, y_predicted_la = compare_test(test_scaled, y_hat_predicted)
print('RMSE Lasso   %.3f' % (rmse))

# LSTM
y_hat_predicted = algorithm.lstm(x_train, y_train, x_test, batch_size=1, nb_epoch=200, neurons=3)
rmse, y_predicted_lstm = compare_test(test_scaled, y_hat_predicted)
print('RMSE LSTM   %.3f' % (rmse))

titles = ['X', 'Y', 'ElasticNet', 'KNN5', 'KNN10', 'SGD', 'Lasso','LSTM']
data = [raw_values[split:-1], y_predicted_real, y_predicted_en, y_predicted_knn5, y_predicted_knn10, y_predicted_sgd,
        y_predicted_la,y_predicted_lstm]
misc.plot_lines_graph('Stationary - Normalization, Test Data ', date[split:], titles, data)
