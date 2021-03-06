from Util import algorithm
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc


def compare_train(train_scaled, y_predicted):
    predictions = list()
    d = raw_values[window_size:split + window_size + 1]
    for i in range(len(y_predicted)):
        X = x_train[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        yhat = data_misc.inverse_difference(d, yhat, len(y_predicted) + 1 - i)
        predictions.append(yhat)

    d = raw_values[window_size + 1:split + window_size + 1]
    rmse = sqrt(mean_squared_error(d, predictions))
    return rmse, predictions


def compare(y_test, y_predicted):
    predictions = list()
    for i in range(len(y_test)):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        d = raw_values[split + window_size - 1:]
        yhat = data_misc.inverse_difference(d, yhat, len(y_test) + 1 - i)
        predictions.append(yhat)

    d = raw_values[split + window_size + 1:]
    rmse = sqrt(mean_squared_error(d, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions


window_size = 15  # 15
series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')
date = series['Date']
series = series.drop(['Date'], axis=1)
date = date.iloc[window_size:]
date = date.values

raw_values = series.values

# Stationary Data
diff_values = data_misc.difference(raw_values, 1)

supervised = data_misc.timeseries_to_supervised(diff_values, window_size)
# print(raw_values)
supervised = supervised.values[window_size:, :]

size_supervised = len(supervised)
split = int(size_supervised * 0.80)

train, test = supervised[0:split], supervised[split:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

print('Size supervised %i' % (size_supervised))

print('------- Train -------')
# No Prediction
y_hat_predicted = y_train
rmse, y_predicted = compare_train(train_scaled, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_hat_predicted = x_train[:, 0]
rmse, y_predicted = compare_train(train_scaled, y_hat_predicted)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_hat_predicted = algorithm.elastic_net2(x_train, y_train, x_train, normalize=False)
rmse, y_predicted = compare_train(train_scaled, y_hat_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 5)
rmse, y_predicted = compare_train(train_scaled, y_hat_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 10)
rmse, y_predicted = compare_train(train_scaled, y_hat_predicted)
print('RMSE KNN(10)  %.3f' % (rmse))

# SGD
y_hat_predicted = algorithm.sgd_regressor(x_train, y_train, x_train)
rmse, y_predicted = compare_train(train_scaled, y_hat_predicted)
print('RMSE SGD     %.3f' % (rmse))

# LSTM
y_hat_predicted = algorithm.lstm(x_train, y_train, x_train, batch_size=1, nb_epoch=3, neurons=1)
rmse, y_predicted = compare_train(train_scaled, y_hat_predicted)
print('RMSE LSTM    %.3f' % (rmse))

print('------- Test --------')
# No Prediction
y_hat_predicted_es = y_test
rmse, y_hat_predicted = compare(y_test, y_hat_predicted_es)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_predicted_dummy_es = x_test[:, 0]
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
y_predicted_lstm = algorithm.lstm(x_train, y_train, x_test, batch_size=1, nb_epoch=100, neurons=15)
rmse, y_predicted_lstm = compare(y_test, y_predicted_lstm)
print('RMSE LSTM    %.3f' % (rmse))

titles = ['Y', 'ElasticNet', 'KNN5', 'KNN10', 'SGD', 'Lasso','LSTM']
data = [y_hat_predicted, y_predicted_en, y_predicted_knn5, y_predicted_knn10, y_predicted_sgd, y_predicted_la, y_predicted_lstm]
# titles = ['Y', 'ElasticNet', 'ElasticNet Future', 'KNN5', 'KNN10', 'SGD']
# data = [y_test, y_predicted_en, y_future_en, y_predicted_knn5, y_predicted_knn10]
# y_future_en = y_future_en[1]
# data = [y_hat_predicted, y_predicted_en, y_future_en, y_predicted_knn5, y_predicted_knn10, y_predicted_sgd]
date_test = date[split + 1:]
print('Length date test:' + str(len(date_test)))
print('Length data test:' + str(len(y_test)))
misc.plot_lines_graph('Stationary - Normalization,Test Data ', date_test, titles, data)
"""
# data = [y_hat_predicted_es, y_predicted_en_es, y_future_en_es, y_predicted_knn5_es, y_predicted_knn10_es,
#        y_predicted_sgd_es]
data = [y_hat_predicted_es, y_predicted_en_es, y_predicted_knn5_es, y_predicted_knn10_es,
        y_predicted_sgd_es, y_predicted_la_sc]

misc.plot_lines_graph('Stationary - Normalization, Test Data ', date_test, titles, data)
"""