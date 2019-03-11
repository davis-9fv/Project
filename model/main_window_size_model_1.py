import datetime
from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from util import data_misc
import numpy as np


def diff_df(df_data):
    processed_columns = []
    for column_name in df_data:
        values_column = df_data[column_name]
        diff_col = data_misc.difference(values_column, 1)
        tmp = []
        for val in diff_col:
            tmp.append(val)

        processed_columns.append(tmp)

    df_diff = DataFrame()
    i = 0
    for column in df_data:
        df_diff[column] = processed_columns[i]
        i = i + 1

    return df_diff


def compare_train(len_y_train=0, y_predicted=[]):
    predictions = list()
    d = avg_values[window_size:split + window_size + 1]
    for i in range(len_y_train):
        X = x_train[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        yhat = data_misc.inverse_difference(d, yhat, len_y_train + 1 - i)
        predictions.append(yhat)

    # the +1 represents the drop that we did when the data was diff
    d = avg_values[window_size + 1:split + window_size + 1]
    rmse = sqrt(mean_squared_error(d, predictions))
    return rmse, predictions


def compare_test(len_y_test=0, y_predicted=[]):
    predictions = list()
    for i in range(len_y_test):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)

        # Stationary
        d = avg_values[split + window_size - 1:]
        yhat = data_misc.inverse_difference(d, yhat, len_y_test + 1 - i)

        predictions.append(yhat)

    # the +1 represents the drop that we did when the data was diff
    d = avg_values[split + window_size + 1:]
    # d = avg_values[split + window_size :]
    rmse = sqrt(mean_squared_error(d, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions



time_start = datetime.datetime.now()
print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
seed = 5
np.random.seed(seed)

result = list()
write_file = True
plot = False

# window_size_opt = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
window_size_opt = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14]
window_size_opt = [3]

lag = 1


# Use of algorithms
use_Dummy = True
use_ElasticNet = True
use_LASSO = True
use_KNN5 = True
use_KNN10 = True
use_SGD = True
use_LSTM = True

normal_train_results = []
dummy_train_results = []
elastic_train_results = []
lasso_train_results = []
knn5_train_results = []
knn10_train_results = []
sgd_train_results = []
lstm_train_results = []
corr_normal_train_results = []
corr_dummy_train_results = []
corr_elastic_train_results = []
corr_lasso_train_results = []
corr_knn5_train_results = []
corr_knn10_train_results = []
corr_sgd_train_results = []
corr_lstm_train_results = []

normal_test_results = []
dummy_test_results = []
elastic_test_results = []
lasso_test_results = []
knn5_test_results = []
knn10_test_results = []
sgd_test_results = []
lstm_test_results = []
corr_normal_test_results = []
corr_dummy_test_results = []
corr_elastic_test_results = []
corr_lasso_test_results = []
corr_knn5_test_results = []
corr_knn10_test_results = []
corr_sgd_test_results = []
corr_lstm_test_results = []

window_size_index = 0


combinations = window_size_opt
total_models = len(combinations)
print('Quantity of Models: %s' % str(total_models))
model_count = 0

path = 'C:/tmp/bitcoin/'
# path = '/code/Project/data/'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_x_size_model_1.csv'

# To pair with the other models, this model gets 1438 first rows.
series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
series = series.iloc[::-1]
avg = series['Avg']
avg_values = avg.values

elasticnet_model = None
lasso_model = None
knn5_model = None
knn10_model = None
sgd_model = None
lstm_model = None

for combination in combinations:
    model_count = model_count + 1
    window_size = combination


    print('----------')
    print(combination)
    print('Model %s / %s' % (str(model_count), str(total_models)))

    # Stationary Data
    # Difference only works for one step. Please implement for other steps
    avg_diff_values = data_misc.difference(avg_values, lag)
    print(avg_diff_values)
    # Avg values converted into supervised model
    avg_supervised = data_misc.timeseries_to_supervised(avg_diff_values, window_size)
    # The first [Window size number] contains zeros which need to be cut.
    avg_supervised = avg_supervised.values[window_size:, :]
    supervised = avg_supervised
    print('Window Size:         %s' % str(window_size))
    print('Lag:                 %s' % str(lag))

    size_supervised = len(supervised)
    split = int(size_supervised * 0.80)
    train, test = supervised[0:split], supervised[split:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = data_misc.scale(train, test)
    print(train_scaled[0:10,:])
    x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
    x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

    total_features = len(train[0])
    print('Total Features: %i' % (total_features))
    print('Total of supervised data: %i' % (size_supervised))

    print(':: Train ::')
    len_y_train = len(y_train)
    # No Prediction
    y_predicted_es = y_train
    rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
    normal_train_results.append(rmse)
    corr_normal_train_results.append(data_misc.correlation(y_train, y_predicted))
    print('RMSE NoPredic  %.3f' % (rmse))

    # Dummy
    if use_Dummy:
        y_hat_predicted_es = x_train[:, -1]
        rmse, y_predicted = compare_train(len_y_train, y_hat_predicted_es)
        dummy_train_results.append(rmse)
        corr_dummy_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE Dummy   %.3f' % (rmse))

    # ElasticNet
    if use_ElasticNet:
        y_predicted_es, elasticnet_model = algorithms.elastic_net(x_train, y_train, x_train, normalize=False)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        elastic_train_results.append(rmse)
        corr_elastic_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE Elastic %.3f' % (rmse))

    # Lasso
    if use_LASSO:
        y_predicted_es, lasso_model = algorithms.lasso(x_train, y_train, x_train, normalize=False)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        lasso_train_results.append(rmse)
        corr_lasso_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE Lasso   %.3f' % (rmse))

    # KNN5
    if use_KNN5:
        y_predicted_es, knn5_model = algorithms.knn_regressor(x_train, y_train, x_train, 3)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        knn5_train_results.append(rmse)
        corr_knn5_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE KNN(5)  %.3f' % (rmse))

    # KNN10
    if use_KNN10:
        y_hat_predicted, knn10_model = algorithms.knn_regressor(x_train, y_train, x_train, 20)
        rmse, y_predicted = compare_train(len_y_train, y_hat_predicted)
        knn10_train_results.append(rmse)
        corr_knn10_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE KNN(10)  %.3f' % (rmse))

    # SGD
    if use_SGD:
        y_predicted_es, sgd_model = algorithms.sgd_regressor(x_train, y_train, x_train)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        sgd_train_results.append(rmse)
        corr_sgd_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE SGD     %.3f' % (rmse))

    # LSTM
    if use_LSTM:
        # rmse, y_predicted = 0, 0
        nb_epoch = 1
        neurons = total_features
        print("Epoch: %i  Neurons: %i" % (nb_epoch, neurons))
        y_predicted_es, lstm_model = algorithms.lstm(x_train, y_train, x_train, batch_size=1, nb_epoch=nb_epoch,
                                                     neurons=neurons)
        # y_predicted_es, lstm_model = algorithms.lstm(x_train, y_train, x_train, batch_size=1, nb_epoch=200, neurons=10)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        lstm_train_results.append(rmse)
        corr_lstm_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE LSTM    %.3f' % (rmse))
        print(lstm_model.summary())

        # rmse, y_predicted = 0, 0
        # lstm_train_results.append(rmse)
        # corr_lstm_train_results.append(data_misc.correlation(y_train, y_predicted))
        # print('RMSE LSTM    %.3f' % (rmse))

    print(':: Test ::')
    len_y_test = len(y_test)
    # Use of the algorithms
    # No Prediction

    y_predicted_normal_es = y_test
    rmse, y_predicted_normal = compare_test(len_y_test, y_predicted_normal_es)
    normal_test_results.append(rmse)
    corr_normal_test_results.append(data_misc.correlation(y_test, y_predicted_normal))
    print('RMSE NoPredic  %.3f' % (rmse))

    if use_Dummy:
        y_predicted_es = x_test[:, -1]
        rmse, y_predicted_dummy = compare_test(len_y_test, y_predicted_es)
        dummy_test_results.append(rmse)
        corr_dummy_test_results.append(data_misc.correlation(y_test, y_predicted_dummy))
        print('RMSE Dummy   %.3f' % (rmse))

    if use_ElasticNet:
        y_predicted_es = elasticnet_model.predict(x_test)
        rmse, y_predicted_en = compare_test(len_y_test, y_predicted_es)
        elastic_test_results.append(rmse)
        corr_elastic_test_results.append(data_misc.correlation(y_test, y_predicted_en))
        print('RMSE Elastic %.3f' % (rmse))

    if use_LASSO:
        y_predicted_es = lasso_model.predict(x_test)
        rmse, y_predicted_la = compare_test(len_y_test, y_predicted_es)
        lasso_test_results.append(rmse)
        corr_lasso_test_results.append(data_misc.correlation(y_test, y_predicted_la))
        print('RMSE Lasso   %.3f' % (rmse))

    if use_KNN5:
        y_predicted_es = knn5_model.predict(x_test)
        rmse, y_predicted_knn5 = compare_test(len_y_test, y_predicted_es)
        knn5_test_results.append(rmse)
        corr_knn5_test_results.append(data_misc.correlation(y_test, y_predicted_knn5))
        print('RMSE KNN(5)  %.3f' % (rmse))

    if use_KNN10:
        y_predicted_es = knn10_model.predict(x_test)
        rmse, y_predicted_knn10 = compare_test(len_y_test, y_predicted_es)
        knn10_test_results.append(rmse)
        corr_knn10_test_results.append(data_misc.correlation(y_test, y_predicted_knn10))
        print('RMSE KNN(10) %.3f' % (rmse))

    if use_SGD:
        y_predicted_es = sgd_model.predict(x_test)
        rmse, y_predicted_sgd = compare_test(len_y_test, y_predicted_es)
        sgd_test_results.append(rmse)
        corr_sgd_test_results.append(data_misc.correlation(y_test, y_predicted_sgd))
        print('RMSE SGD     %.3f' % (rmse))

    if use_LSTM:
        # rmse, y_predicted = 0, 0
        y_predicted_es = algorithms.lstm_predict(lstm_model, x_test, batch_size=1)
        rmse, y_predicted_lstm = compare_test(len_y_test, y_predicted_es)
        lstm_test_results.append(rmse)
        corr_lstm_test_results.append(data_misc.correlation(y_test, y_predicted_lstm))
        print('RMSE LSTM    %.3f' % (rmse))

    print('----------')

# Get results
results_df = DataFrame({
    '(Tr) Normal': normal_train_results,
    '(Tr) Normal Corr': corr_normal_train_results,
    '(Tr) Dummy': dummy_train_results,
    '(Tr) Dummy Corr': corr_dummy_train_results,
    '(Tr) ElasticNet': elastic_train_results,
    '(Tr) ElasticNet Corr': corr_elastic_train_results,
    '(Tr) Lasso': lasso_train_results,
    '(Tr) Lasso Corr': corr_lasso_train_results,
    '(Tr) KNN5': knn5_train_results,
    '(Tr) KNN5 Corr': corr_knn5_train_results,
    '(Tr) KNN10': knn10_train_results,
    '(Tr) KNN10 Corr': corr_knn10_train_results,
    '(Tr) SGD': sgd_train_results,
    '(Tr) SGD Corr': corr_sgd_train_results,
    '(Tr) LSTM': lstm_train_results,
    '(Tr) LSTM Corr': corr_lstm_train_results,

    '(Te) Normal': normal_test_results,
    '(Te) Normal Corr': corr_normal_test_results,
    '(Te) Dummy': dummy_test_results,
    '(Te) Dummy Corr': corr_dummy_test_results,
    '(Te) ElasticNet': elastic_test_results,
    '(Te) ElasticNet Corr': corr_elastic_test_results,
    '(Te) Lasso': lasso_test_results,
    '(Te) Lasso Corr': corr_lasso_test_results,
    '(Te) KNN5': knn5_test_results,
    '(Te) KNN5 Corr': corr_knn5_test_results,
    '(Te) KNN10': knn10_test_results,
    '(Te) KNN10 Corr': corr_knn10_test_results,
    '(Te) SGD': sgd_test_results,
    '(Te) SGD Corr': corr_sgd_test_results,
    '(Te) LSTM': lstm_test_results,
    '(Te) LSTM Corr': corr_lstm_test_results})

col_win_size = []
col_lag = []
col_use_bitcoin = []
col_bitcoin_data = []
col_use_trend = []
col_trend_data = []

for combination in combinations:
    col_win_size.append(combination)

results_df['Window Size'] = col_win_size


print(results_df)

if write_file:
    results_df.to_csv(path + output_file, header=True)

time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
