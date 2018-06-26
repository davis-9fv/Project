from sklearn.neighbors import KNeighborsRegressor
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
from Util import algorithm


def compare_train(train_scaled, y_predicted):
    predictions = list()
    for i in range(len(train_scaled)):
        X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        # Se agrega split+1 ya para que sea consecuente con la data para realizar el diff.
        yhat = data_misc.inverse_difference(raw_values[0:split + 1], yhat, len(train_scaled) + 1 - i)
        # print(yhat)
        predictions.append(yhat)

    # Se empieza desde uno ya que el primer dato no se puede tomar en cuenta por diff.
    rmse = sqrt(mean_squared_error(raw_values[1:split + 1], predictions))
    return rmse


def compare_test(test_scaled, y_predicted):
    predictions = list()
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        yhat = data_misc.inverse_difference(raw_values[split:], yhat, len(test_scaled) +1 - i)
        # print(yhat)
        predictions.append(yhat)

    # Se aumenta uno ya que el primer dato no se puede tomar en cuenta por diff.
    rmse = sqrt(mean_squared_error(raw_values[1 + split:], predictions))
    return rmse


series = read_csv('../Thesis/Bitcoin_historical_data_processed_1f.csv', header=0, sep='\t')

# transform data to be stationary
raw_values = series['Avg'].values
size_raw_values = len(raw_values)
split = int(size_raw_values * 0.80)

print('Raw values total size: %i, Train size: %i, Test size: %i ' % (size_raw_values, split, size_raw_values - split))

date = series['Date'].values
diff_values = data_misc.difference(raw_values, 1)

supervised = data_misc.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:split], supervised_values[split:]

# para que solo funcione para pruebas
# train, test = diff_values[0:-10], diff_values[-10:]
# train = train.values.reshape(train.shape[0], 1)
# test = test.values.reshape(test.shape[0], 1)

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

#y_predicted = train_scaled[:, 1]
#print(compare_train(train_scaled, y_predicted))
#y_predicted = test_scaled[:, 1]
#print(compare_test(test_scaled, y_predicted))

x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

print('------- Train -------')
# No Prediction
y_predicted = train_scaled[:, 1]
rmse = compare_train(train_scaled, y_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# ElasticNet
y_predicted = algorithm.elastic_net(x_train, y_train, x_train)
rmse = compare_train(train_scaled, y_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 5)
rmse = compare_train(train_scaled, y_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_predicted = algorithm.knn_regressor(x_train, y_train, x_train, 10)
rmse = compare_train(train_scaled, y_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# SGD
y_predicted = algorithm.sgd_regressor(x_train, y_train, x_train)
rmse = compare_train(train_scaled, y_predicted)
print('RMSE SGD     %.3f' % (rmse))

print('------- Test --------')
# No Prediction
y_predicted = test_scaled[:, 1]
rmse = compare_test(test_scaled, y_predicted)
print('RMSE NoPredic  %.3f' % (rmse))

# ElasticNet
y_predicted = algorithm.elastic_net(x_train, y_train, x_test)
rmse = compare_test(test_scaled, y_predicted)
print('RMSE Elastic %.3f' % (rmse))

# KNN5
y_predicted = algorithm.knn_regressor(x_train, y_train, x_test, 5)
rmse = compare_test(test_scaled, y_predicted)
print('RMSE KNN(5)  %.3f' % (rmse))

# KNN10
y_predicted = algorithm.knn_regressor(x_train, y_train, x_test, 10)
rmse = compare_test(test_scaled, y_predicted)
print('RMSE KNN(10) %.3f' % (rmse))

# SGD
y_predicted = algorithm.sgd_regressor(x_train, y_train, x_test)
rmse = compare_test(test_scaled, y_predicted)
print('RMSE SGD     %.3f' % (rmse))

# for i in range(len(test_scaled)):
#    inversed = scaler.inverse_transform([test_scaled[i]])
#    yhat = data_misc.inverse_difference(raw_values[-11:], inversed, len(test_scaled) + 1 - i)
#    print(yhat)

# for i in range(len(train_scaled)):
#    inversed = scaler.inverse_transform([train_scaled[i]])
#    yhat = data_misc.inverse_difference(raw_values[:-10], inversed, len(train_scaled) + 1 - i)
#    print(yhat)
