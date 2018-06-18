# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
from sklearn.model_selection import train_test_split


def train_test_split(data, test_size=0.20):
    rows = data.shape[0]
    split = int(rows * test_size)
    train = data[0:-split, :]
    test = data[-split:rows]
    return train, test


def x_y_split(data):
    columns = data.shape[1]
    x = data[:, 0:columns - 1]
    y = data[:, columns - 1:columns]
    return x, y


# scale train and test data to [-1, 1]
def scale(train, test):
    # train the normalization
    scaler = StandardScaler()
    scaler = scaler.fit(train)
    # Reshape the data
    train = train.reshape(train.shape[0], train.shape[1])
    test = test.reshape(test.shape[0], test.shape[1])
    # Transform the data
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(X, y):
    # print(X)
    # print(X.shape)
    # From vertical we reshape to horizontal
    X = X.reshape(1, X.shape[0])
    # print(X)
    # print(X.shape)
    # We stack the array and the value
    y_inverted = np.column_stack((X, y))
    y_inverted = scaler.inverse_transform(y_inverted)
    # We get the last value of the array
    y_inverted = y_inverted[0, -1]
    return y_inverted


# fit an LSTM network to training data
def fit_lstm(X_train, y_train, batch_size, nb_epoch, neurons):
    # print(X_train.shape)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    # print(X_train.shape[0])
    # print(X_train.shape[1])
    # print(X_train.shape[2])
    # print(X_train.shape)
    y_train = y_train.reshape(y_train.shape[0], 1)

    activ_func = 'tanh'
    dropout = 0.25

    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
                   stateful=True, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons - 5, return_sequences=True, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons - 2, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(Activation(activ_func))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # print(model.summary())
    for i in range(nb_epoch):
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X_test):
    X_test = X_test.reshape(1, 1, X_test.shape[0])
    y_hat = model.predict(X_test, batch_size=batch_size)
    return y_hat[0, 0]


# np.random.seed(seed=9)
series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')
raw_data = series.values

raw_train, raw_test = train_test_split(raw_data, 0.20)
scaler, train_scaled, test_scaled = scale(raw_train, raw_test)
X_train, y_train = x_y_split(train_scaled)
X_test, y_test = x_y_split(test_scaled)

print(train_scaled)

# repeat experiment
repeats = 1
error_scores = list()
for r in range(repeats):
    # fit the model
    lstm_model = fit_lstm(X_train, y_train, 1, nb_epoch=10, neurons=10)
    predictions = list()
    for i in range(len(X_test)):
        X, y = X_test[i], y_test[i]
        yhat = forecast_lstm(lstm_model, 1, X)
        # print("Yhat: " + str(yhat) + " Y_test: " + str(y))
        yhat_raw = invert_scale(X, yhat)
        print("Yhat: " + str(yhat_raw) + " Y_test: " + str(raw_test[i, -1]))
        # print("---------------")
        predictions.append(yhat_raw)
    # report performance

    rmse = sqrt(mean_squared_error(raw_test[:, -1], predictions))
    print('%d) Test RMSE: %.3f' % (r + 1, rmse))
    error_scores.append(rmse)

    pyplot.plot(raw_test[:, -1])
    pyplot.plot(predictions, color='red')
    pyplot.show()

results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
