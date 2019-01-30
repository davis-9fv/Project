# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy
from Util import misc
from Util import data_misc


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # print(X)
    model = Sequential()
    # print(X.shape[1])
    # print(X.shape[2])

    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    # model.add(Dense(6, activation='relu'))
    # model.add(Dense(6, activation='linear'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# load dataset
series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')
# series = read_csv('data/airline-passengers.csv', header=0, sep='\t')

numpy.random.seed(seed=9)

date = series['Date'].values
window_size = 12
raw_values = series['Passangers'].values
size_raw_values = len(raw_values)
split = int(size_raw_values * 0.80)  # 0.80
print('raw_values:' + str(len(raw_values)))

diff_values = data_misc.difference(raw_values, 1)

# transform data to be supervised learning
supervised = data_misc.timeseries_to_supervised(diff_values, window_size)
# We cut because the head is full of zeroes. We cut the lenght of the window size
supervised_values = supervised.values[window_size:, :]

# split data into train and test-sets
train, test = supervised_values[:split], supervised_values[split:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

# repeat experiment
repeats = 10
error_scores = list()

nb_epoch = 200
neurons = 10
print("N° Epoch: %i   N° Neurons:  %i" % (nb_epoch, neurons))
for r in range(repeats):
    # fit the model
    lstm_model = fit_lstm(train_scaled, batch_size=1, nb_epoch=nb_epoch, neurons=neurons)
    # forecast the entire training dataset to build up state for forecasting
    # train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    # lstm_model.predict(train_reshaped, batch_size=1)
    # walk-forward validation on the test data
    predictions = list()
    normal_y = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        x_train, y_train = test_scaled[i, 0:-1], test_scaled[i, -1]

        # Forecast for test
        y_predicted = forecast_lstm(lstm_model, 1, x_train)
        # invert scaling
        y_predicted = data_misc.invert_scale(scaler, x_train, y_predicted)
        y = data_misc.invert_scale(scaler, x_train, y_train)

        # invert differencing test
        # We get the history and we cut because we pair with the supervised that was cut before. It was full of zeroes
        d = raw_values[split + window_size - 1:]
        y_predicted = data_misc.inverse_difference(d, y_predicted, len(test_scaled) + 1 - i)
        y = data_misc.inverse_difference(d, y, len(test_scaled) + 1 - i)

        # print(" Y_test: " + str(y) + " Yhat: " + str(yhat) + " yraw:" + str(raw_values[i + len(train) + 1]))
        # store forecast
        normal_y.append(y)
        predictions.append(y_predicted)

    # report performance
    # the +1 represents the drop that we did when the data was diff
    y_raw = raw_values[split + 1 + window_size:]
    test_rmse = sqrt(mean_squared_error(y_raw, predictions))
    print('%d) Test RMSE: %.3f' % (r + 1, test_rmse))
    # print(predictions)
    error_scores.append(test_rmse)
    # plot
    misc.plot_line_graph2('LSTM_rmse_' + str(test_rmse), date[split + 1 + window_size::], y_raw, predictions)

# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
# misc.plot_data_graph2('Data', date, raw_values)
# results.boxplot()
# pyplot.show()
