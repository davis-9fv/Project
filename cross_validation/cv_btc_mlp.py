#!/usr/bin/env python
import itertools
from pandas import read_csv
from util import data_misc
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler

from sklearn import neural_network
import numpy as np
import pandas as pd
import os
import config as conf
from threading import Thread
from time import sleep
import datetime
from sklearn.preprocessing import StandardScaler
import config


class MyThread(Thread):
    window_size = 0
    hl1 = 0
    hl2 = 0
    optimization = ''
    activation = ''
    coefs = []
    rmse_val = 0
    rmse_test = 0
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None
    y_test = None

    def run(self):
        print("{} started!".format(self.getName()))  # "Thread-x started!"

        print("Train VS Val")
        nn = neural_network.MLPRegressor(solver=self.optimization,
                                         batch_size='auto',
                                         max_iter=1000000000,
                                         shuffle=False, early_stopping=True)

        nn.activation = self.activation
        if self.hl2 == 0:
            nn.hidden_layer_sizes = (self.hl1,)
        else:
            nn.hidden_layer_sizes = (self.hl1, self.hl2)
        y_val_predicted_list = []
        y_train_val_predicted_list = []
        y_test_predicted_list = []

        nn.fit(self.x_train, self.y_train)
        y_val_predicted = nn.predict(self.x_val)
        y_val_predicted_list.append(nn.predict(self.x_val))
        rmse = sqrt(mean_squared_error(self.y_val, y_val_predicted))
        self.rmse_val = rmse
        print('RMSE Val MLP   %.3f   ' % (rmse))

        print("Train + Val VS Test")
        x_train_val = np.concatenate((x_train, x_val), axis=0)
        y_train_val = np.concatenate((y_train, y_val), axis=0)

        nn.fit(x_train_val, y_train_val)
        y_test_predicted = nn.predict(x_test)
        rmse = sqrt(mean_squared_error(y_test, y_test_predicted))
        self.rmse_test = rmse
        print('RMSE Test MLP   %.3f  ' % (rmse))
        print("{} finished!".format(self.getName()))

        # print(self.hl1)
        # print(self.hl2)
        # print(self.optimization)
        # print(self.window_size)
        # print(self.rmse_val)
        # print(self.rmse_test)

        df = pd.DataFrame({'hl1': self.hl1,
                           'hl2': self.hl2,
                           'Optimization': self.optimization,
                           'Activation': self.activation,
                           'WindowSize': self.window_size,
                           'RMSE Val': self.rmse_val,
                           'RMSE Test': self.rmse_test
                           }, index=[0])

        filename = conf.selected_path + "CV" + "_" + conf.output_file_mlp
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', sep=',', header=False)
        else:
            df.to_csv(filename, sep=',', header=True)


time_start = datetime.datetime.now()
print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
hidden_layer_1 = [5, 10, 50, 100, 150, 200, 250]
hidden_layer_2 = [0, 5, 10, 50, 100, 150, 200]
# hidden_layer_2 = [1]
optimization = ['sgd', 'adam']
# optimization = ['sgd']
# alphas = [0.0001, 0.001, 0.01]
activation = ['relu', 'tanh']

combinations = [hidden_layer_1,
                hidden_layer_2,
                optimization,
                activation]
# combinations = [window_size,lag]
combinations = list(itertools.product(*combinations))
print(combinations)
print(len(combinations))

windows_sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#windows_sizes = [4, 5]

for window_size in windows_sizes:

    print("Window Size %i" % (window_size))
    # window_size = 5  # 15
    path = config.selected_path
    input_file = config.input_file_ds
    series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
    series = series.iloc[::-1]

    for i in range(0, 30):
        corr = series['Avg'].autocorr(lag=i)
        print('Corr: %.2f Lang: %i' % (corr, i))

    avg = series['Avg']
    avg_values = avg.values
    # Stationary Data
    # diff_values = data_misc.difference(avg_values, 1)
    # avg_values = diff_values

    print("Diff values")

    supervised = data_misc.timeseries_to_supervised(avg_values, window_size)
    # print(raw_values)
    supervised = supervised.values[window_size:, :]
    # supervised = list(range(1, 101))

    size_supervised = len(supervised)
    split_train_val = int(size_supervised * 0.60)
    split_val_test = int(size_supervised * 0.20)

    train = supervised[0:split_train_val]
    val = supervised[split_train_val:split_train_val + split_val_test]
    test = supervised[split_train_val + split_val_test:]

    scaler = StandardScaler()
    scaler = scaler.fit(train)
    # scaler = StandardScaler()
    # scaler.fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    x_train, y_train = train[:, 0:-1], train[:, -1]
    x_val, y_val = val[:, 0:-1], val[:, -1]
    x_test, y_test = test[:, 0:-1], test[:, -1]

    print('LSTM - BTC')
    print('Window Size %i' % (window_size))
    print('Size Train %i' % (len(train)))
    print('Size Val %i' % (len(val)))
    print('Size Test %i' % (len(test)))

    print('Size supervised %i' % (size_supervised))
    print('Size raw_values %i' % (len(avg_values)))

    # alphas = np.linspace(12, 0, 50)
    alphas = np.linspace(10, 0, 100)
    print(alphas)
    # print("Total Alphas")
    # print(len(alphas))

    """"
    """
    thread_list = []
    for y in range(len(combinations)):
        # for y in range(1):
        mythread = MyThread(name="Thread-{}".format(y + 1))  # ...Instantiate a thread and pass a unique ID to it
        mythread.x_train = x_train
        mythread.y_train = y_train
        mythread.x_val = x_val
        mythread.y_val = y_val
        mythread.x_test = x_test
        mythread.y_test = y_test
        combination = combinations[y]
        mythread.hl1 = combination[0]
        mythread.hl2 = combination[1]
        mythread.optimization = combination[2]
        mythread.activation = combination[3]
        mythread.window_size = window_size

        thread_list.append(mythread)
        # mythread.isAlive()
    # mythread.start()  # ...Start the thread

    num_threads_running = 50
    num_buckets = int(len(thread_list) / num_threads_running) + 1
    print("NUmber of buckets %i" % (num_buckets))

    for bucket in range(num_buckets):
        running_threads = []
        process_bucket = True
        print("---------Loading Bucket %i" % (bucket))
        for y in range(num_threads_running):
            position = (bucket * num_threads_running) + y
            if position < len(thread_list):
                print("---------Position %i" % (position))
                running_threads.append(thread_list[position])
                thread = thread_list[position]
                thread.start()

        while process_bucket:
            print("---------Processing Bucket %i" % (bucket))
            for y in range(len(running_threads)):
                thread = running_threads[y]
                process_bucket = process_bucket * (not thread.isAlive())

            if not process_bucket:
                print("---------Sleep Starts")
                sleep(30)
                print("---------Sleep Ends")

        print("---------Finishing Bucket %i" % (bucket))

time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
