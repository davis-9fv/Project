from sklearn.metrics import mean_squared_error
from math import sqrt
from util import data_misc
import util.data_creation as dc


def compare_train(window_size, scaler, len_y_train, x_train, y_predicted=[]):
    predictions = list()
    d = dc.avg_values[window_size:dc.split_train_test + window_size + 1]
    for i in range(len_y_train):
        X = x_train[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        yhat = data_misc.inverse_difference(d, yhat, len_y_train + 1 - i)
        predictions.append(yhat)

    # the +1 represents the drop that we did when the data was diff
    d = dc.avg_values[window_size + 1:dc.split_train_test + window_size + 1]
    rmse = sqrt(mean_squared_error(d, predictions))
    return rmse, predictions


def compare_val(window_size, scaler, len_y_val, x_val, y_predicted=[]):
    predictions = list()
    for i in range(len_y_val):
        X = x_val[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)

        # Stationary
        #print(yhat)
        # Limit between the train and the val. the val d works from the end to the start.
        d = dc.avg_values[dc.split_train_val + window_size:dc.split_train_val + window_size + 1 + len_y_val]
        yhat = data_misc.inverse_difference(d, yhat, len_y_val + 1 - i)

        predictions.append(yhat)

    #print('Predictions')
    #print(predictions)

    # the +1 represents the drop that we did when the data was diff
    d = dc.avg_values[dc.split_train_val + window_size + 1: dc.split_train_val + window_size + 1 + dc.split_val_test]
    # d = avg_values[split + window_size :]
    rmse = sqrt(mean_squared_error(d, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions


def compare_test(window_size, scaler, len_y_test, x_test, y_predicted=[]):
    predictions = list()
    for i in range(len_y_test):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)

        # Stationary
        d = dc.avg_values[dc.split_train_test + window_size - 1:]
        yhat = data_misc.inverse_difference(d, yhat, len_y_test + 1 - i)

        predictions.append(yhat)

    # the +1 represents the drop that we did when the data was diff
    d = dc.avg_values[dc.split_train_test + window_size + 1:]
    # d = avg_values[split + window_size :]
    rmse = sqrt(mean_squared_error(d, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions
