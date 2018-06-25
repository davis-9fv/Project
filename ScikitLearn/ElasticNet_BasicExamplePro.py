# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
from sklearn.linear_model import ElasticNet
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
from pandas import DataFrame
from pandas import Series

series = read_csv('../Thesis/Bitcoin_historical_data_processed_1f.csv', header=0, sep='\t')

# transform data to be stationary
raw_values = series['Avg'].values
diff_values = data_misc.difference(raw_values, 1)

# transform data to be supervised learning
supervised = data_misc.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets ------------
train, test = supervised_values[0:-1], supervised_values[-1:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], 1))
print(X_train.shape)
print(X_test.shape)
X_test = X_test.reshape(X_test.shape[0], 1)
print(X_test.shape)

regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)

print(regr.score(X_train, y_train))
print(regr.coef_)
print(regr.intercept_)

print('y_test: ')
print(y_test)
# print('y_predicted: ')
# print(y_predicted)

predictions = list()
allList = list()

for i in range(len(raw_values)):
    allList.append(raw_values[i])

# Se pone esta por que se va a calcular el valor futuro y ya esta presente
allList = allList[:-1]
flag = False

for i in range(20):
    X, y = 0, 0
    if (flag):
        diff_values = data_misc.difference(raw_values, 1)
        supervised = data_misc.timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values
        train, test = supervised_values[0:-1], supervised_values[-1:]
        test_scaled = scaler.transform(test)
        X, y = test_scaled[0, 0:-1], test_scaled[0, -1]
    else:
        flag = True
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]

    yhat = regr.predict([X])
    print("Y_test: " + str(y) + " Yhat: " + str(yhat))

    yhat = data_misc.invert_scale(scaler, X, yhat)

    # Se recorre -1 porque para que no se alinie donde empezó
    yhat = data_misc.inverse_difference(raw_values, yhat, -1 - i)
    # store forecast
    predictions.append(yhat)
    allList.append(yhat)

    # df = DataFrame(raw_values)
    # print(df[0][i])
    # columns = [df[0][i] for i in range(0,df.size)]
    # columns.append(yhat)
    raw_values = allList
    # print(columns)

rmse = sqrt(mean_squared_error(raw_values[-20:], predictions))
print('Test RMSE: %.7f' % (rmse))
misc.plot_line_graph('ElasticNet', raw_values[-20:], predictions)
misc.print_comparison_list('Raw', raw_values[-20:], predictions)

misc.plot_data_graph('Data', raw_values)
misc.plot_data_graph('All data', allList)

misc.plot_data_graph('RawData', raw_values[:-20])
