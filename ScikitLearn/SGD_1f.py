# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
from sklearn import linear_model
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

series = read_csv('../tests/shampoo-sales3.csv', header=0)
raw_data = series.values
X_train, X_test, y_train, y_test = train_test_split(raw_data[:, 0], raw_data[:, 1], test_size=0.33, random_state=9)

X_train = X_train.reshape(X_train.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], 1)

clf = linear_model.SGDRegressor(max_iter=2000)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

print('y_test: ')
print(y_test)
print('y_predicted: ')
print(y_predicted)

rmse = sqrt(mean_squared_error(y_test, y_predicted))
print('Test RMSE: %.7f' % (rmse))
