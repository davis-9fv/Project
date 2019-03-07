# https://machinelearningmastery.com/moving-average-smoothing-for-time-series-forecasting-python/

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import Series
from pandas import unique
import numpy
from pandas import concat
from numpy import mean
from sklearn.preprocessing import StandardScaler


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


# scale train and test data to [-1, 1]
def standarize(train, test):
    # fit scaler
    scaler = StandardScaler()
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


# inverse scaling for a forecasted value
def invert_scale_array(scaler, X, values):
    new_row = [x for x in X]
    for i in range(0, len(values)):
        new_row = new_row + [values[i]]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -len(values):]


# create a differenced series
# Trabaja perfectamente, se come el primer valor de la data
def difference(dataset, interval=1, include_first_item=False):
    diff = list()
    if include_first_item:
        diff.append(0)
    for i in range(interval, len(dataset)):
        num1 = dataset[i]
        num2 = dataset[i - interval]

        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
# Trabaja perfectamente, elimina el primer valor de la data
def inverse_difference(history, yhat, interval=1):
    value = history[-interval]
    result = yhat + value
    return result


def inverse_difference2(first_raw_element, train_diff, yhat):
    sum_train_diff = sum(float(i) for i in train_diff)
    result = yhat + sum_train_diff + first_raw_element
    return result


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(lag, 0, -1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    # print(df)
    return df


def from_scaled_diff_raw(scaler, ):
    print('')


def timeseries_to_moving_average(data, window=3):
    # window = 3
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


def get_train_length(dataset, batch_size, test_percent):
    # substract test_percent to be excluded from training, reserved for testset
    length = len(dataset)
    length *= 1 - test_percent
    train_length_values = []
    for x in range(int(length) - 100, int(length)):
        modulo = x % batch_size
        if (modulo == 0):
            train_length_values.append(x)
            print(x)
    return (max(train_length_values))


def get_test_length(dataset, batch_size, upper_train, timesteps):
    test_length_values = []
    for x in range(len(dataset) - 200, len(dataset) - timesteps * 2):
        modulo = (x - upper_train) % batch_size
        if (modulo == 0):
            test_length_values.append(x)
            print(x)
    return (max(test_length_values))


# Creating a data structure with n timesteps
def data_to_timesteps(train, length, timesteps):
    X_train = []
    y_train = []
    print(length + timesteps)
    for i in range(timesteps, length + timesteps):
        X_train.append(train[i - timesteps:i])
        y_train.append(train[i:i + timesteps])

    print(len(X_train))
    print(len(y_train))
    print(numpy.array(X_train).shape)
    print(numpy.array(y_train).shape)

    X_train, y_train = numpy.array(X_train), numpy.array(y_train)
    X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = numpy.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

    return X_train, y_train


# Creating a data structure with n timesteps
def data_to_timesteps(train, length, timesteps):
    X_train = []
    y_train = []
    print(length + timesteps)
    for i in range(timesteps, length + timesteps):
        X_train.append(train[i - timesteps:i])
        y_train.append(train[i:i + timesteps])

    print(len(X_train))
    print(len(y_train))
    print(numpy.array(X_train).shape)
    print(numpy.array(y_train).shape)

    X_train, y_train = numpy.array(X_train), numpy.array(y_train)
    X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = numpy.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

    return X_train, y_train


# Creating a data structure with n timesteps
def test_data_to_timesteps(test, testset_length, timesteps):
    X_test = []
    y_test = []
    for i in range(timesteps, testset_length + timesteps):
        X_test.append(test[i - timesteps:i])
        y_test.append(test[i:i + timesteps])
    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)
    X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = numpy.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    return X_test, y_test


# Convert categorical features to numerical binary features
def cat_to_num(data):
    categories = unique(data)
    features = []
    for cat in categories:
        binary = (data == cat)
        features.append(binary.astype("int"))
    return features


def slide_data(data, lag=2):
    data_previous = data
    data = data[lag:]
    data_previous = data_previous[:-lag]
    return data, data_previous


def correlation(col1, col2):
    df = DataFrame({'col1': col1, 'col2': col2})
    corr_matrix = df.corr(method='pearson', min_periods=1)
    result = corr_matrix.iloc[0, 1]
    return result
