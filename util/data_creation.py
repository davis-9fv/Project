from pandas import read_csv
from util import data_misc
import config
import numpy as np
from pandas import DataFrame

global split_train_test
split_train_test = 0
global split_train_val
split_train_val = 0
global split_val_test
split_val_test = 0

global avg_values
avg_values = []


def load_dataset():
    path = config.selected_path
    input_file = config.input_file_ds

    # To pair with the other models, this model gets 1438 first rows.
    series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
    series = series.iloc[::-1]
    return series


def create_avg_supervised_ds(window_size, series, cut_window_size):
    avg = series['Avg']
    global avg_values
    avg_values = avg.values
    lag = 1

    # Stationary Data
    # Difference only works for one step. Please implement for other steps
    avg_diff_values = data_misc.difference(avg_values, lag)
    # Avg values converted into supervised model
    avg_supervised = data_misc.timeseries_to_supervised(avg_diff_values, window_size)
    # The first [Window size number] contains zeros which need to be cut.
    if cut_window_size:
        avg_supervised = avg_supervised.values[window_size:, :]

    supervised = avg_supervised

    print('AVG Window Size:     %s' % str(window_size))
    print('Lag:                 %s' % str(lag))

    return supervised


def create_btc_supervised_ds(window_size, series, columns):
    df = DataFrame()
    for column in columns:
        df[column] = series[column]

    processed_columns = []
    for column_name in df:
        values_column = df[column_name]
        diff_col = data_misc.difference(values_column, 1)
        supervised_col = data_misc.timeseries_to_supervised(diff_col, window_size)
        # We drop the last column from weight_supervised because it is not the target we want
        supervised_col = supervised_col.values[:, :-1]
        processed_columns.append(supervised_col)

    flag = True
    supervised = []
    for col in processed_columns:
        if flag:
            supervised = col
            flag = False
        else:
            supervised = np.concatenate((supervised, col), axis=1)

    print('BTC Window Size:     %s' % str(window_size))
    print('Columns:             %s' % str(columns))

    return supervised


def split_ds_train_test(supervised):
    size_supervised = len(supervised)
    global split_train_test
    split_train_test = int(size_supervised * config.train_rate)
    train, test = supervised[0:split_train_test], supervised[split_train_test:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = data_misc.scale(train, test)
    x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
    x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]
    return scaler, x_train, y_train, x_test, y_test


def split_ds_train_val_test(supervised):
    size_supervised = len(supervised)
    global split_train_val
    split_train_val = int(size_supervised * config.train_val_rate)
    global split_val_test
    split_val_test = int(size_supervised * config.val_test_rate)
    global split_train_test
    split_train_test = split_train_val + split_val_test
    train = supervised[0:split_train_val]
    val = supervised[split_train_val:split_train_val + split_val_test]
    test = supervised[split_train_val + split_val_test:]

    # transform the scale of the data
    scaler, train_scaled, val_scaled = data_misc.scale(train, val)
    scaler, train_scaled, test_scaled = data_misc.scale(train, test)

    # print(val[:, -1])
    x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
    x_val, y_val = val_scaled[:, 0:-1], val_scaled[:, -1]
    x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

    return scaler, x_train, y_train, x_val, y_val, x_test, y_test


def get_avg_values():
    return avg_values
