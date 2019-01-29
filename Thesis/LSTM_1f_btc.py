# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from tests.models import Sequential
from tests.layers import Dense, Activation
from tests.layers import LSTM
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
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
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
series = read_csv('../Thesis/Bitcoin_historical_data_processed_1f.csv', header=0, sep='\t')

numpy.random.seed(seed=9)

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

# repeat experiment
repeats = 20
error_scores = list()
for r in range(repeats):
    # fit the model
    lstm_model = fit_lstm(train_scaled, 10, nb_epoch=3, neurons=1)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # yhat = y
        # invert scaling
        yhat = data_misc.invert_scale(scaler, X, yhat)
        y = data_misc.invert_scale(scaler, X, y)

        # invert differencing
        yhat = data_misc.inverse_difference(raw_values, yhat, len(test_scaled) + 0 - i)
        y = data_misc.inverse_difference(raw_values, y, len(test_scaled) + 0 - i)

        # print(" Y_test: " + str(y) + " Yhat: " + str(yhat) + " yraw:" + str(raw_values[i + len(train) + 1]))
        # store forecast
        predictions.append(yhat)

    # report performance

    rmse = sqrt(mean_squared_error(raw_values[-365:], predictions))
    print('%d) Test RMSE: %.3f' % (r + 1, rmse))
    # print(predictions)
    error_scores.append(rmse)

    # plot
    misc.plot_line_graph2('LSTM_rmse_' + str(rmse), date[-365:], raw_values[-365:], predictions)

# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
misc.plot_data_graph2('Data', date, raw_values)
# results.boxplot()
# pyplot.show()
