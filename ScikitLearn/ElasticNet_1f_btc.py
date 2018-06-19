# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
from sklearn.linear_model import ElasticNet
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy
from Util import misc


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    # print(df)
    return df


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')
# Si se pone shuffle true, se mejora la respuesta
# X_train, X_test, y_train, y_test = train_test_split(raw_data[:, 3], raw_data[:, 4], test_size=0.20, shuffle=False)

# transform data to be stationary
raw_values = series['Avg'].values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-365], supervised_values[-365:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

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

print(regr.coef_)
print(regr.intercept_)

y_predicted = regr.predict(X_test)

print('y_test: ')
print(y_test)
print('y_predicted: ')
print(y_predicted)

predictions = list()

for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = y_predicted[i]
    print("Y_test: " + str(y) + " Yhat: " + str(yhat))

    yhat = invert_scale(scaler, X, yhat)
    print("yhat no scaled:" + str(yhat))
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    print("yhat no difference:" + str(yhat))
    # store forecast
    predictions.append(yhat)

rmse = sqrt(mean_squared_error(raw_values[-365:], predictions))
print('Test RMSE: %.7f' % (rmse))


misc.plot_line_graph('ElasticNet', raw_values[-365:], predictions)
