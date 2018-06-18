import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


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
def scale_standar_scaler(train, test):
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


series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t', nrows=10)
raw_data = series.values

train, test = train_test_split(raw_data, test_size=0.20)

scaler, train_scaled, test_scaled = scale_standar_scaler(train, test)

print("---------")
print(train)
print("---------")
print(train_scaled)
print("---------")
print(scaler.inverse_transform(train_scaled[7]))
print("---------")
x, y = x_y_split(test)
print(y)
