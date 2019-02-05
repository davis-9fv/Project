from pandas import read_csv
from Util import data_misc
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import L1L2
from pandas import DataFrame


# https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/

def lstm(x_train, y_train, x_to_predict, batch_size, nb_epoch=3, neurons=3, l1l2=[]):
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    model = Sequential()
    l1 = l1l2[0]
    l2 = l1l2[1]
    model.add(LSTM(neurons,
                   bias_regularizer=L1L2(l1=l1, l2=l2),
                   batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),
                   stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    for i in range(nb_epoch):
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

    y_predicted = list()
    for i in range(len(x_to_predict)):
        X = x_to_predict[i]
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        y_predicted.append(yhat[0, 0])

    return y_predicted, model


def lstm_predict(model, x_to_predict, batch_size):
    y_predicted = list()
    for i in range(len(x_to_predict)):
        X = x_to_predict[i]
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        y_predicted.append(yhat[0, 0])

    return y_predicted


def experiment(neurons, epochs, alpha, n_repeats):
    rmse_val = []
    rmse_test = []
    result = []
    print("----")
    print("Neurons: %i      Epochs: %i      NRepeats: %i" % (neurons, epochs, n_repeats))
    for r in range(n_repeats):
        np.random.seed(r)
        print(":: Repeat %i/%i" % (r + 1, n_repeats))
        print('L1: %s   L2: %s' % (alpha[0], alpha[1]))
        print("Train VS Val")
        y_val_predicted, lstm_model = lstm(x_train, y_train, x_val,
                                           neurons=neurons,
                                           nb_epoch=epochs,
                                           batch_size=1,
                                           l1l2=alpha)
        rmse = sqrt(mean_squared_error(y_val, y_val_predicted))
        rmse_val.append(rmse)
        print('     RMSE LSTM   %.3f' % (rmse))

        print("Train + Val VS Test")
        x_train_val = np.concatenate((x_train, x_val), axis=0)
        y_train_val = np.concatenate((y_train, y_val), axis=0)

        y_test_predicted, lstm_model = lstm(x_train_val, y_train_val, x_test,
                                            neurons=neurons,
                                            nb_epoch=epochs,
                                            batch_size=1,
                                            l1l2=alpha)
        rmse = sqrt(mean_squared_error(y_test, y_test_predicted))
        rmse_test.append(rmse)
        print('     RMSE LSTM   %.3f' % (rmse))

    rmse_total = np.add(rmse_val, rmse_test)
    rmse_total = np.divide(rmse_total, 2)
    rmse_total_avg = np.average(rmse_total)

    rmse_val_avg = np.average(rmse_val)
    rmse_test_avg = np.average(rmse_test)

    # result.append()
    return [rmse_val_avg, rmse_test_avg, rmse_total_avg, neurons, epochs, alpha[0], alpha[1], n_repeats]


window_size = 5  # 15
#path = '/code/Project/data/'
path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'cv_btc_lstm_results.csv'
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

scaler = MinMaxScaler(feature_range=(0, 1))
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

l1 = np.linspace(3, -3, 1)
l1_ = []
for val in l1:
    l1_.append(val)

l1_.append(0)

l2 = np.linspace(3, -3, 1)
l2_ = []
for val in l2:
    l2_.append(val)

l2_.append(0)

neurons_list = [1]
epochs_list = [1]
alphas_list = [l1_, l2_]
alphas_list = list(itertools.product(*alphas_list))
n_repeats = 3

print(alphas_list)
print("Total Alphas")
print(len(alphas_list))

coefs = []
rmse_val = []
rmse_test = []
overal_result = []

print("Configuration:::")
print("neurons_list:    %s" % (neurons_list))
print("epochs_list:     %s" % (epochs_list))
print("alphas_list:     %s" % (alphas_list))

for neurons in neurons_list:
    print("\n\n")
    print("Working on Neuron %i" % (neurons))
    for epochs in epochs_list:

        print("Working on epoch %i" % (epochs))
        for alpha in alphas_list:
            print("Working on alpha %s %s" % (alpha[0], alpha[1]))
            overal_result.append(experiment(neurons, epochs, alpha, n_repeats))
            # print("\n")

        # print("\n")

"""
overal_result = [[4.8764619190322449, 2.4759713560382588, 3.6762166375352519, 3, 1, 3.0, 3.0, 1],
                 [4.6192575154574298, 3.8625729371934523, 4.2409152263254413, 3, 1, 3.0, -3.0, 1],
                 [4.5857459737531592, 3.3139867957911857, 3.9498663847721724, 3, 1, 3.0, 0, 1],
                 [4.361134948713028, 5.2231896113362311, 4.7921622800246295, 3, 1, -3.0, 3.0, 1],
                 [4.556261005491244, 3.8712109642443178, 4.2137359848677809, 3, 1, -3.0, -3.0, 1],
                 [4.4938886920089951, 3.8707193725099822, 4.1823040322594887, 3, 1, -3.0, 0, 1],
                 [4.5407651877522612, 5.2241070326740626, 4.8824361102131615, 3, 1, 0, 3.0, 1],
                 [4.556261124206582, 2.3840437129432792, 3.4701524185749308, 3, 1, 0, -3.0, 1],
                 [4.3724195912159578, 5.2265501206219431, 4.7994848559189505, 3, 1, 0, 0, 1],
                 [4.1797180954466784, 5.1854127499349074, 4.6825654226907929, 4, 1, 3.0, 3.0, 1],
                 [4.5262174255019652, 6.2180292821734326, 5.3721233538376989, 4, 1, 3.0, -3.0, 1],
                 [4.8447743979300384, 6.7370763483692313, 5.7909253731496353, 4, 1, 3.0, 0, 1],
                 [4.9316743550580568, 3.8854037032696467, 4.408539029163852, 4, 1, -3.0, 3.0, 1],
                 [4.464320456382012, 5.2238482338172956, 4.8440843450996542, 4, 1, -3.0, -3.0, 1],
                 [4.471561587971534, 6.2085963527135437, 5.3400789703425389, 4, 1, -3.0, 0, 1],
                 [3.9194352526078653, 6.2109827153660397, 5.0652089839869525, 4, 1, 0, 3.0, 1],
                 [4.4938699065978165, 5.2237586080898399, 4.8588142573438287, 4, 1, 0, -3.0, 1],
                 [4.1340378936483999, 3.635274839866077, 3.8846563667572385, 4, 1, 0, 0, 1]]
"""
print(overal_result)
columns = ['rmse_val_avg', 'rmse_test_avg', 'rmse_total_avg', 'neurons', 'epochs', 'alpha[0]',
           'alpha[1]', 'n_repeats']
df_results = DataFrame(overal_result, columns=columns)
df_results.to_csv(path + output_file, header=True)
