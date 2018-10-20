from Util import algorithm
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
from sklearn.preprocessing import StandardScaler

def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


for x in range(12, 13):
    window_size = x  # 15
    print('Window Size: %i' %(x))
    series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')
    date = series['Date']
    series = series.drop(['Date'], axis=1)
    date = date.iloc[window_size:]
    date = date.values

    raw_values = series.values
    raw_values = data_misc.timeseries_to_supervised(raw_values, window_size)
    # print(raw_values)
    raw_values = raw_values.values[window_size:, :]

    size_raw_values = len(raw_values)
    split = int(size_raw_values * 0.80)

    train, test = raw_values[0:split], raw_values[split:]
    # fit transform
    transformer = StandardScaler()
    transformer.fit(train)
    train = transformer.transform(train)
    test = transformer.transform(test)
    print(train)
    print(test)

    x_train, y_train = train[:, 0:-1], train[:, -1]
    x_test, y_test = test[:, 0:-1], test[:, -1]





    print('Size raw_values %i' % (size_raw_values))

    print('------- Train -------')
    # No Prediction
    y_hat_predicted = y_test
    rmse = compare(y_test, y_hat_predicted)
    print('RMSE NoPredic  %.3f' % (rmse))

    # Dummy
    y_predicted_dummy = x_train[:, 0]
    rmse = compare(y_train, y_predicted_dummy)
    print('RMSE Dummy   %.3f' % (rmse))

    # ElasticNet
    y_predicted_en = algorithm.elastic_net2(x_train, y_train, x_train)
    rmse = compare(y_train, y_predicted_en)
    print('RMSE Elastic %.3f' % (rmse))

    print('------- Test --------')
    # No Prediction
    y_hat_predicted = y_test
    rmse = compare(y_test, y_hat_predicted)
    print('RMSE NoPredic  %.3f' % (rmse))

    # Dummy
    y_predicted_dummy = x_test[:, 0]
    rmse = compare(y_test, y_predicted_dummy)
    print('RMSE Dummy   %.3f' % (rmse))

    # ElasticNet
    y_predicted_en, y_future_en = algorithm.elastic_net(x_train, y_train, x_test, y_test)
    rmse = compare(y_test, y_predicted_en)
    print('RMSE Elastic %.3f' % (rmse))
    rmse = compare(y_test, y_future_en)
    print('RMSE Elastic Future %.3f' % (rmse))
    print('  ')



    titles = ['Y test', 'ElasticNet', 'ElasticNet Future']
    data = [y_test, y_predicted_en, y_future_en]
    date_test = date[split:]
    misc.plot_lines_graph('Raw Data, Test Data, Window size ' + str(x), date_test, titles, data)
