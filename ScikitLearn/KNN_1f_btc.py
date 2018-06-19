from sklearn.neighbors import KNeighborsRegressor
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from math import sqrt
from matplotlib import pyplot
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

n_neighbors = 10
neigh = KNeighborsRegressor(algorithm='kd_tree', leaf_size=30, weights='uniform', n_neighbors=n_neighbors, n_jobs=4)
neigh.fit(X_train, y_train)

predictions = list()

y_predicted = neigh.predict(X_test)

# raw_values[-365:]

for i in range(len(y_predicted)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = y_predicted[i]
    print("Scaled: Y_test: " + str(y) + " Yhat: " + str(yhat))

    yhat = invert_scale(scaler, X, yhat)
    print("yhat no scaled:" + str(yhat))

    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    print("yhat no difference:" + str(yhat))
    predictions.append(yhat)

rmse = sqrt(mean_squared_error(raw_values[-365:], predictions))
print('Test RMSE: %.7f' % (rmse))

y = raw_values[-365:]

misc.print_comparison_list('RawData', y, predictions)
misc.plot_line_graph('KNN(' + str(n_neighbors) + ')', raw_values[-365:], predictions)
misc.plot_line_graph('KNN(' + str(n_neighbors) + ')', raw_values[-365:], predictions)
