# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet

from sklearn import linear_model
from sklearn.linear_model import ElasticNet
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from matplotlib import pyplot

supervised = True
if supervised:
    series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')
    raw_data = series.values
    # Si se pone shuffle true, se mejora la respuesta
    X_train, X_test, y_train, y_test = train_test_split(raw_data[:, 3], raw_data[:, 4], test_size=0.20, shuffle=False)
else:
    series = read_csv('../Thesis/Bitcoin_historical_data_processed.csv', header=0, sep='\t')
    series['index'] = 0
    series['index'] = [i for i in range(0, series.shape[0])]
    raw_data = series.values
    # Si se pone shuffle true, se mejora la respuesta
    X_train, X_test, y_train, y_test = train_test_split(raw_data[:, 4], raw_data[:, 3], test_size=0.20, shuffle=False)

print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], 1))
X_test = X_test.reshape(X_test.shape[0], 1)
print(X_train.shape)

regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)

print(regr.coef_)
print(regr.intercept_)

y_predicted = regr.predict(X_test)

print('y_test: ')
print(y_test)
print('y_predicted: ')
print(y_predicted)

rmse = sqrt(mean_squared_error(y_test, y_predicted))
print('Test RMSE: %.7f' % (rmse))

pyplot.plot(y_test)
pyplot.plot(y_predicted, color='red')
pyplot.show()
