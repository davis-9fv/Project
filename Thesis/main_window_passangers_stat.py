from Util import algorithm
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc


def compare_train(y_test, y_predicted):
    predictions = list()
    d = raw_values[window_size:split + window_size + 1]
    for i in range(len(y_train)):
        yhat = y_predicted[i]
        yhat = data_misc.inverse_difference(d, yhat, len(y_train) + 1 - i)
        predictions.append(yhat)

    d = raw_values[window_size + 1:split + window_size + 1]
    rmse = sqrt(mean_squared_error(d, predictions))
    return rmse, predictions


def compare_test(y_test, y_predicted):
    predictions = list()
    for i in range(len(y_test)):
        y = y_test[i]
        yhat = y_predicted[i]
        d = raw_values[split + window_size:]
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
x_train, y_train = train[:, 0:-1], train[:, -1]
x_test, y_test = test[:, 0:-1], test[:, -1]

print('Size supervised %i' % (size_supervised))

print('------- Train -------')
# No Prediction
y_hat_predicted = y_train
rmse, y_predicted = compare_train(x_test, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_hat_predicted = x_train[:, 0]
rmse, y_predicted = compare_train(x_test, y_hat_predicted)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_hat_predicted = algorithm.elastic_net2(x_train, y_train, x_train, normalize=False)
rmse, y_predicted = compare_train(x_test, y_hat_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 5)
rmse, y_predicted = compare_train(x_test, y_hat_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_hat_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 10)
rmse, y_predicted = compare_train(x_test, y_hat_predicted)
print('RMSE KNN(10)  %.3f' % (rmse))

# SGD
y_hat_predicted = algorithm.sgd_regressor(x_train, y_train, x_train)
rmse, y_predicted = compare_train(x_test, y_hat_predicted)
print('RMSE SGD     %.3f' % (rmse))

# LSTM
y_hat_predicted = algorithm.lstm(x_train, y_train, x_train, batch_size=1, nb_epoch=3, neurons=1)
rmse, y_predicted = compare_train(x_test, y_hat_predicted)
print('RMSE LSTM    %.3f' % (rmse))

print('------- Test --------')
# No Prediction
y_hat_predicted = y_test
rmse, y_hat_predicted = compare_test(y_test, y_hat_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_predicted_dummy_es = x_test[:, 0]
rmse, y_predicted_dummy = compare_test(y_test, y_predicted_dummy_es)
print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_predicted_en_es, y_future_en_es = algorithm.elastic_net(x_train, y_train, x_test, y_test, normalize=False)
rmse, y_predicted_en = compare_test(y_test, y_predicted_en_es)
print('RMSE Elastic %.3f' % (rmse))

# y_future_en = compare_test(y_test, y_future_en_es)

# KNN5
y_predicted_knn5_es = algorithm.knn_regressor(x_train, y_train, x_test, 5)
rmse, y_predicted_knn5 = compare_test(y_test, y_predicted_knn5_es)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_predicted_knn10_es = algorithm.knn_regressor(x_train, y_train, x_test, 10)
rmse, y_predicted_knn10 = compare_test(y_test, y_predicted_knn10_es)
print('RMSE KNN(10) %.3f' % (rmse))

# SGD
y_predicted_sgd_es = algorithm.sgd_regressor(x_train, y_train, x_test)
rmse, y_predicted_sgd = compare_test(y_test, y_predicted_sgd_es)
print('RMSE SGD     %.3f' % (rmse))

# Lasso
y_predicted_la_es = algorithm.lasso(x_train, y_train, x_test, normalize=False)
rmse, y_predicted_la = compare_test(y_test, y_predicted_la_es)
print('RMSE Lasso    %.3f' % (rmse))

# titles = ['Y', 'ElasticNet', 'ElasticNet Future', 'KNN5', 'KNN10']
# y_future_en = y_future_en[1]
# data = [y_hat_predicted, y_predicted_en, y_future_en, y_predicted_knn5, y_predicted_knn10]

titles = ['Y', 'ElasticNet', 'KNN5', 'KNN10', 'SGD', 'Lasso']
data = [y_hat_predicted, y_predicted_en, y_predicted_knn5, y_predicted_knn10, [], y_predicted_la]

date_test = date[split + 1:]
print('Length date test:' + str(len(date_test)))
print('Length data test:' + str(len(y_test)))

misc.plot_lines_graph('Stationary, Test Data ', date_test, titles, data)

data = [y_test, y_predicted_en_es, y_future_en_es, y_predicted_knn5_es, y_predicted_knn10_es]
misc.plot_lines_graph('Stationary, Test Data ', date_test, titles, data)

"""
y = list()
y_1 = list()
y_2 = list()
y_3 = list()
y_4 = list()

for i in range(len(y_train)):
    y.append(float(y_train[i]))
    y_1.append(float(y_train[i]))
    y_2.append(float(y_train[i]))
    y_3.append(float(y_train[i]))
    y_4.append(float(y_train[i]))

for i in range(len(y_test)):
    y.append(float(y_test[i]))
    y_1.append(float(y_predicted_en_es[i]))
    y_2.append(float(y_future_en_es[i]))
    y_3.append(float(y_predicted_knn5_es[i]))
    y_4.append(float(y_predicted_knn10_es[i]))

titles = ['Y', 'ElasticNet', 'ElasticNet Future', 'KNN5', 'KNN10']
data = [y, y_1, y_2, y_3, y_4]
misc.plot_lines_graph('Stationary, Test Data ', date, titles, data)
"""
