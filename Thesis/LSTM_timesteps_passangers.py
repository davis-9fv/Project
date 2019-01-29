# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from tests.layers import Input
from tests.models import Model
import numpy
from Util import misc
from Util import data_misc
from tests.models import Sequential
from tests.layers import Dense, Activation
from tests.layers import LSTM


def compare(y_test, y_predicted):
    predictions = list()
    for i in range(len(y_test)):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = sc.inverse_transform(yhat)
        yhat = yhat.flatten()
        d = raw_values[upper_train:]

        yhat = data_misc.inverse_difference(d, yhat[0], len(y_test) + 1 - i)
        predictions.append(yhat)

    z = raw_values[upper_train :upper_train+testset_length]

    rmse = sqrt(mean_squared_error(z, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions


# load dataset
series = read_csv('../data/airline-passengers.csv', header=0, sep='\t')

numpy.random.seed(seed=9)

date = series['Date'].values
date = numpy.delete(date, (-1), axis=0)
series['Passangers'].to_csv('../data/out.csv',  index = False)

raw_values = series['Passangers'].values
print(len(raw_values))
diff_values = data_misc.difference(raw_values, 1)

batch_size = 10
nb_epoch = 300
length = data_misc.get_train_length(dataset=diff_values, batch_size=batch_size, test_percent=0.20)
print(length)

timesteps = 3

upper_train = length + timesteps * 2
df_data_1_train = diff_values[0:upper_train]
training_set = df_data_1_train.values
# reshaping
training_set = numpy.reshape(training_set, (training_set.shape[0], 1))

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set = sc.fit_transform(numpy.float64(training_set))
print(training_set.shape)

X_train, y_train = data_misc.data_to_timesteps(train=training_set, length=length, timesteps=timesteps)
print(X_train.shape)
print(y_train.shape)

# model = Sequential()
# print(X.shape[1])
# print(X.shape[2])

# model.add(LSTM(4, batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]), stateful=True))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# for i in range(nb_epoch):
#    model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
#    model.reset_states()

inputs_1_mae = Input(batch_shape=(batch_size, timesteps, 1))
lstm_1_mae = LSTM(10, stateful=True, return_sequences=True)(inputs_1_mae)
lstm_2_mae = LSTM(10, stateful=True, return_sequences=True)(lstm_1_mae)

output_1_mae = Dense(units=1)(lstm_2_mae)

regressor_mae = Model(inputs=inputs_1_mae, outputs=output_1_mae)

regressor_mae.compile(optimizer='adam', loss='mae')
regressor_mae.summary()

epochs = 2
for i in range(epochs):
    print("Epoch: " + str(i))
    regressor_mae.fit(X_train, y_train, shuffle=False, epochs=1, batch_size=batch_size)
    regressor_mae.reset_states()

test_length = data_misc.get_test_length(dataset=raw_values, batch_size=batch_size, upper_train=upper_train,
                                        timesteps=timesteps)
print(test_length)
upper_test = test_length + timesteps * 2
testset_length = test_length - upper_train
print(testset_length)

print(upper_train, upper_test, len(raw_values))

# df_data_1_test = raw_values[upper_train - 1:upper_test]
df_data_1_test = diff_values[upper_train:upper_test]
test_set = df_data_1_test.values

# reshaping
test_set = numpy.reshape(test_set, (test_set.shape[0], 1))

# scaling
test_set = sc.fit_transform(numpy.float64(test_set))

x_test, y_test = data_misc.test_data_to_timesteps(test=test_set, testset_length=testset_length, timesteps=timesteps)

rmse, predictions = compare(y_test, y_test)

predicted_bcg_values_test_mae = regressor_mae.predict(x_test, batch_size=batch_size)
regressor_mae.reset_states()

print(predicted_bcg_values_test_mae.shape)

# reshaping
predicted_bcg_values_test_mae = numpy.reshape(predicted_bcg_values_test_mae,
                                              (predicted_bcg_values_test_mae.shape[0],
                                               predicted_bcg_values_test_mae.shape[1]))

y_test = numpy.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

print(predicted_bcg_values_test_mae.shape)

# inverse transform
predicted_bcg_values_test_mae = sc.inverse_transform(predicted_bcg_values_test_mae)
y_test = sc.inverse_transform(y_test)

# creating y_test data
y_hat_1 = []
y_hat_2 = []
y_hat_3 = []
for j in range(0, testset_length - timesteps):
    y_hat_1 = numpy.append(y_hat_1, predicted_bcg_values_test_mae[j, 0])
    y_hat_2 = numpy.append(y_hat_2, predicted_bcg_values_test_mae[j, 1])
    y_hat_3 = numpy.append(y_hat_3, predicted_bcg_values_test_mae[j, 2])

# reshaping
y_hat_1 = y_hat_1.flatten()
y_hat_2 = y_hat_2.flatten()
y_hat_3 = y_hat_3.flatten()

y_test_1 = y_test[:, 0]
y_test_2 = y_test[:, 1]
y_test_3 = y_test[:, 2]
y_test_1 = y_test_1.flatten()
y_test_2 = y_test_2.flatten()
y_test_3 = y_test_3.flatten()

misc.plot_line_graph2('LSTM_rmse_1', date[-20:], y_hat_1, y_hat_1)
misc.plot_line_graph2('LSTM_rmse_2', date[-20:], y_hat_2, y_hat_2)
misc.plot_line_graph2('LSTM_rmse_3', date[-20:], y_hat_3, y_hat_3)
