# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
from sklearn import linear_model
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

supervised = True
if supervised:
    series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')
    raw_data = series.values
    X_train, X_test, y_train, y_test = train_test_split(raw_data[:, 0:4], raw_data[:, 4], test_size=0.20, shuffle=True)
else:
    series = read_csv('../Thesis/Bitcoin_historical_data_processed.csv', header=0, sep='\t')
    raw_data = series.values
    X_train, X_test, y_train, y_test = train_test_split(raw_data[:, 0:3], raw_data[:, 3], test_size=0.20, shuffle=False)

print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1])
print(X_train.shape)

clf = linear_model.SGDRegressor(max_iter=1000, verbose=True, shuffle=False)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

print('y_test: ')
print(y_test)
print('y_predicted: ')
print(y_predicted)

rmse = sqrt(mean_squared_error(y_test, y_predicted))
print('Test RMSE: %.7f' % (rmse))
