from builtins import super

from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc

series = read_csv('../Thesis/Bitcoin_historical_data_processed_1f.csv', header=0, sep='\t')

# transform data to be stationary
raw_values = series['Avg'].values
date = series['Date'].values
#raw_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#raw_values = [0, 2, 6, 36, 8, 10, 18, 84, 16, 18, 30, 132, 24, 26, 168, 45, 32, 136, 54, 57, 160]

size_raw_values = len(raw_values)
split = int(size_raw_values * 0.50) #994



diff_values = data_misc.difference(raw_values, 1)

# transform data to be supervised learning
supervised = data_misc.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:split], supervised_values[split:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

y_predicted = x_test

predictions = list()
for i in range(len(x_test)):
    X, y = x_test[i], y_test[i]
    yhat = y_predicted[i]
    # print("Y_test: " + str(y) + " Yhat: " + str(yhat))

    yhat = data_misc.invert_scale(scaler, X, yhat)
    # print("yhat no scaled:" + str(yhat))

    yhat = data_misc.inverse_difference(raw_values, yhat, len(x_test) +1 - i)
    # print("yhat no difference:" + str(yhat))
    # store forecast

    predictions.append(yhat)

d = raw_values[split+1:]

rmse = sqrt(mean_squared_error(d, predictions))
print('Test RMSE: %.7f' % (rmse))
