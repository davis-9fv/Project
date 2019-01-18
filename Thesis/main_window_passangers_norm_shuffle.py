from Util import misc
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import data_misc
from sklearn.utils import shuffle
import datetime
from Util import algorithm


def compare_test(y_test, y_predicted):
    predictions = list()
    for i in range(len(y_predicted)):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale_array(scaler, X, yhat)
        predictions.append(yhat)

    #d = raw_values[split + window_size:]
    #rmse = sqrt(mean_squared_error(y_test, y_predicted))
    rmse = 0
    return rmse, predictions

time_start = datetime.datetime.now()
shuffle_data = False
window_size = 15  # 15

print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
print('Shuffle: %i' % (shuffle_data))
print('Window Size: %i' % (window_size))

series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')
date = series['Date']
series = series.drop(['Date'], axis=1)
date = date.iloc[window_size:]
date = date.values

raw_values = series.values
supervised = data_misc.timeseries_to_supervised(raw_values, window_size)
# print(raw_values)
print(supervised.head(10))
df = supervised[supervised.columns[0]]
print(df)
supervised = supervised.values[window_size:, :]



if shuffle_data:
    supervised = shuffle(supervised)

size_supervised = len(supervised)
split = int(size_supervised * 0.80)

train, test = supervised[0:split], supervised[split:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

# train_scaled, test_scaled = train, test

x_train, y_train = train_scaled[:, 0:10], train_scaled[:, 10:16]
x_test, y_test = test_scaled[:, 0:10], test_scaled[:, 10:16]

print('Size supervised %i' % (size_supervised))

print('------- Test --------')
# No Prediction
y_hat_predicted_sc = y_test
rmse, y_hat_predicted = compare_test(y_test, y_hat_predicted_sc)
#y_test = y_hat_predicted
#print('RMSE NoPredic  %.3f' % (rmse))

# Dummy
y_predicted_dummy_sc = x_test[:, 0]
#rmse, y_predicted_dummy = compare_test(y_test, y_predicted_dummy_sc)
#print('RMSE Dummy   %.3f' % (rmse))

# ElasticNet
y_predicted_en_sc = algorithm.elastic_net2(x_train, y_train, x_test, normalize=False)
rmse, y_predicted_en = compare_test(y_test, y_predicted_en_sc)
print('RMSE Elastic %.3f' % (rmse))

# rmse, y_future_en = compare_test(y_test, y_future_en_sc)
# print('RMSE Fut Els %.3f' % (rmse))

# KNN5
#y_predicted_knn5_sc = algorithm.knn_regressor(x_train, y_train, x_test, 5)
#rmse, y_predicted_knn5 = compare_test(y_test, y_predicted_knn5_sc)
#print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
#y_predicted_knn10_sc = algorithm.knn_regressor(x_train, y_train, x_test, 10)
#rmse, y_predicted_knn10 = compare_test(y_test, y_predicted_knn10_sc)
#print('RMSE KNN(10) %.3f' % (rmse))

# SGD
#y_predicted_sgd_sc = algorithm.sgd_regressor(x_train, y_train, x_test)
#rmse, y_predicted_sgd = compare_test(y_test, y_predicted_sgd_sc)
#print('RMSE SGD     %.3f' % (rmse))

# Lasso
y_predicted_la_sc = algorithm.lasso(x_train, y_train, x_test, normalize=False)
rmse, y_predicted_la = compare_test(y_test, y_predicted_la_sc)
print('RMSE Lasso   %.3f' % (rmse))

# LSTM
#y_predicted_lstm = algorithm.lstm(x_train, y_train, x_test, batch_size=1, nb_epoch=200, neurons=3)
#rmse, y_predicted_lstm = compare_test(y_test, y_predicted_lstm)
#print('RMSE LSTM    %.3f' % (rmse))

# print('Y_test')
# print(y_test)

for i in range(0,len(y_hat_predicted)):
    titles = ['Y', 'ElasticNet', 'Lasso']
    data = [y_hat_predicted[i], y_predicted_en[i], y_predicted_la[i]]
    # titles = ['', 'Y', 'ElasticNet', 'KNN5', 'KNN10', 'SGD']
    # data = [[], y_test, y_predicted_en, y_predicted_knn5, y_predicted_knn10, y_predicted_sgd]
    date_test = date[split:]
    print('Length date test:' + str(len(date_test)))
    print('Length data test:' + str(len(y_test)))

    misc.plot_lines_graph('Normalization, Test Data ', date_test, titles, data)

"""
y = list()
y_1 = list()
y_2 = list()
for i in range(len(y_train)):
    y.append(float(y_train[i]))
    y_1.append(float(y_train[i]))
#    y_2.append(float(y_train[i]))

for i in range(len(y_test)):
    y.append(float(y_test[i]))
    y_1.append(float(y_predicted_en[i]))
#    y_2.append(float(y_future_en[i]))

titles = ['Y', 'ElasticNet', 'ElasticNet Future']
data = [y, y_1, y_2]
misc.plot_lines_graph('Normalization, Test Data ', date, titles, data)

# print(y)
"""
