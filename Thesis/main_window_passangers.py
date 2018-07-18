from Util import algorithm
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc


def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


window_size = 15  # 15
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

# KNN5
y_predicted_knn5 = algorithm.knn_regressor(x_train, y_train, x_train, 5)
rmse = compare(y_train, y_predicted_knn5)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_predicted_knn10 = algorithm.knn_regressor(x_train, y_train, x_train, 10)
rmse = compare(y_train, y_predicted_knn10)
print('RMSE KNN(10) %.3f' % (rmse))

# SGD
y_predicted_sgd = algorithm.sgd_regressor(x_train, y_train, x_train)
rmse = compare(y_train, y_predicted_sgd)
print('RMSE SGD     %.3f' % (rmse))

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

# KNN5
y_predicted_knn5 = algorithm.knn_regressor(x_train, y_train, x_test, 5)
rmse = compare(y_test, y_predicted_knn5)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_predicted_knn10 = algorithm.knn_regressor(x_train, y_train, x_test, 10)
rmse = compare(y_test, y_predicted_knn10)
print('RMSE KNN(10) %.3f' % (rmse))

# SGD
y_predicted_sgd = algorithm.sgd_regressor(x_train, y_train, x_test)
rmse = compare(y_test, y_predicted_sgd)
print('RMSE SGD     %.3f' % (rmse))

# print('Y_test')
# print(y_test)

titles = [' ', 'Y', 'ElasticNet', 'KNN5', 'KNN10']
data = [[], y_test, y_predicted_en, y_predicted_knn5, y_predicted_knn10]

#titles = [' ', 'Y', 'ElasticNet', 'ElasticNet Future', 'KNN5', 'KNN10']
#data = [[], y_test, y_predicted_en, y_future_en, y_predicted_knn5, y_predicted_knn10]


date_test = date[split:]
print('Length date test:' + str(len(date_test)))
print('Length data test:' + str(len(y_test)))

misc.plot_lines_graph('Raw Data, Test Data ', date_test, titles, data)

y = list()
y_1 = list()
y_2 = list()
for i in range(len(y_train)):
    y.append(float(y_train[i]))
    y_1.append(float(y_train[i]))
    y_2.append(float(y_train[i]))

for i in range(len(y_test)):
    y.append(float(y_test[i]))
    y_1.append(float(y_predicted_en[i]))
    y_2.append(float(y_future_en[i]))

titles = ['Y', 'ElasticNet', 'ElasticNet Future']
data = [y, y_1, y_2]
# misc.plot_lines_graph('Models, Test Data ', date, titles, data)

# print(y)
