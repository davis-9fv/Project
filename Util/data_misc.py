# https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import Series
import numpy
from pandas import concat
from numpy import mean


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
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


# create a differenced series
# Trabaja perfectamente, se come el primer valor de la data
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
# Trabaja perfectamente, se come el primer valor de la data
def inverse_difference(history, yhat, interval=1):
    value = history[-interval]
    result = yhat + value
    return result


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    # print(df)
    return df


def from_scaled_diff_raw(scaler, ):
    print('')


def timeseries_to_moving_average(data, window=3):
    window = 3
    history = [data[i] for i in range(window)]
    test = [data[i] for i in range(window, len(data))]
    predictions = list()
    # walk forward over time steps in test
    for t in range(len(test)):
        length = len(history)
        yhat = mean([history[i] for i in range(length - window, length)])
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)

    return predictions
