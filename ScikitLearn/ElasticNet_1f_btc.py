# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
from sklearn.linear_model import ElasticNet
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc

series = read_csv('../Thesis/Bitcoin_historical_data_processed_1f.csv', header=0, sep='\t')

# transform data to be stationary
raw_values = series['Avg'].values
date = series['Date'].values
diff_values = data_misc.difference(raw_values, 1)

# transform data to be supervised learning
supervised = data_misc.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-365], supervised_values[-365:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

# print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], 1))
# print(X_train.shape)
# print(X_test.shape)
X_test = X_test.reshape(X_test.shape[0], 1)
# print(X_test.shape)

regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)

# print(regr.score(X_train, y_train))
# print(regr.coef_)
# print(regr.intercept_)

y_predicted = regr.predict(X_test)

print('y_test: ')
print(y_test)
print('y_predicted: ')
print(y_predicted)

predictions = list()
for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = y_predicted[i]
    # print("Y_test: " + str(y) + " Yhat: " + str(yhat))

    yhat = data_misc.invert_scale(scaler, X, yhat)
    # print("yhat no scaled:" + str(yhat))

    yhat = data_misc.inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # print("yhat no difference:" + str(yhat))
    # store forecast
    predictions.append(yhat)

rmse = sqrt(mean_squared_error(raw_values[-365:], predictions))
print('Test RMSE: %.7f' % (rmse))
misc.plot_line_graph2('ElasticNet', date[-365:], raw_values[-365:], predictions)
misc.plot_data_graph2('Data', date, raw_values)
misc.print_comparison_list('Raw', raw_values[-365:], predictions)


