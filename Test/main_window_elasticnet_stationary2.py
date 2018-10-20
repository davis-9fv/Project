from Util import algorithm
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc


def compare(y_raw_test, y_predicted):
    predictions = list()
    predictions2 = list()
    train_tmp = list()
    for i in range(len(train)):
        train_tmp.append(train[i, -1])

    sum_first_values = raw_values[2]
    for i in range(len(y_predicted)):
        yhat = y_predicted[i]
        yhat = data_misc.inverse_difference(raw_values, yhat, len(y_test) + 1 - i)
        predictions.append(yhat)

        a = data_misc.inverse_difference2(sum_first_values, train_tmp, y_predicted[i])
        train_tmp.append(y_predicted[i])
        predictions2.append(a)

    rmse = sqrt(mean_squared_error(y_raw_test, predictions))
    rmse2 = sqrt(mean_squared_error(y_raw_test, predictions2))
    print('predictions2: %i'%(rmse2))
    return rmse, predictions


for x in range(2, 3):
    window_size = x  # 15
    diff_size = 1
    print('Window Size: %i' % (window_size))
    print('Diff Size: %i' % (diff_size))
    #series = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

    #series = read_csv('../data/airline-passengers_20rows.csv', header=0, sep='\t')
    series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')

    date = series['Date']
    series = series.drop(['Date'], axis=1)
    date = date.iloc[window_size:]
    date = date.values

    raw_values = series.values
    #raw_values = series
    # Stationary Data
    diff_values = data_misc.difference(raw_values, diff_size)

    supervised_values = data_misc.timeseries_to_supervised(diff_values, window_size)
    # print(raw_values)
    supervised_values = supervised_values.values[window_size:, :]

    size_supervised_values = len(supervised_values)
    split = int(size_supervised_values * 0.80)

    train, test = supervised_values[0:split], supervised_values[split:]
    x_train, y_train = train[:, 0:-1], train[:, -1]
    x_test, y_test = test[:, 0:-1], test[:, -1]

    y_raw_test = raw_values[split + diff_size + window_size:len(raw_values)]

    print('Size raw_values %i' % (size_supervised_values))



    print('------- Test --------')
    # No Prediction
    y_hat_predicted = y_test
    rmse, y_hat_predicted = compare(y_raw_test, y_hat_predicted)
    print('RMSE NoPredic  %.3f' % (rmse))

    # Dummy
    y_predicted_dummy = x_test[:, 0]
    rmse, y_predicted_dummy = compare(y_raw_test, y_predicted_dummy)
    print('RMSE Dummy   %.3f' % (rmse))

    # ElasticNet
    y_predicted_en, y_future_en = algorithm.elastic_net(x_train, y_train, x_test, y_test)
    rmse, y_predicted_en = compare(y_raw_test, y_predicted_en)
    print('RMSE Elastic %.3f' % (rmse))
    rmse, y_future_en = compare(y_raw_test, y_future_en)
    print('RMSE Elastic Future %.3f' % (rmse))
    print('  ')

    y = list()
    y_1 = list()
    y_2 = list()
    y_3 = list()
    y_4 = list()

    # transform to 1D for showing graph
    for i in range(len(y_raw_test)):
        y.append(float(y_raw_test[i]))
        y_1.append(float(y_predicted_en[i]))
        y_2.append(float(y_future_en[i]))

    #titles = ['Y test', 'ElasticNet', 'ElasticNet Future']
    #data = [y, y_1, y_2]
    #date_test = date[split:-diff_size]
    #misc.plot_lines_graph('Raw Data, Test Data, Window size ' + str(x), date_test, titles, data)
