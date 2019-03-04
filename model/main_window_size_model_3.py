import datetime
from util import algorithms
from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from util import data_misc
import numpy as np
import itertools
import btc_columns
from pandas import RangeIndex


def supervised_diff_dt(df_data, window_size):
    processed_columns = []
    for column_name in df_data:
        values_column = df_data[column_name]
        diff_col = data_misc.difference(values_column, 1)
        supervised_col = data_misc.timeseries_to_supervised(diff_col, window_size)
        # We drop the last column from weight_supervised because it is not the target we want
        supervised_col = supervised_col.values[:, :-1]
        processed_columns.append(supervised_col)

    flag = True
    result = []
    for col in processed_columns:
        if flag:
            result = col
            flag = False
        else:
            result = np.concatenate((result, col), axis=1)

    return result


def compare_train(len_y_train=0, y_predicted=[]):
    predictions = list()
    d = avg_values[total_window_size:split + total_window_size + 1]
    for i in range(len_y_train):
        X = x_train[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)
        yhat = data_misc.inverse_difference(d, yhat, len_y_train + 1 - i)
        predictions.append(yhat)

    d = avg_values[total_window_size + 1:split + total_window_size + 1]
    rmse = sqrt(mean_squared_error(d, predictions))
    return rmse, predictions


def compare_test(len_y_test=0, y_predicted=[]):
    predictions = list()
    for i in range(len_y_test):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)

        # Stationary
        d = avg_values[split + total_window_size - 1:]
        yhat = data_misc.inverse_difference(d, yhat, len_y_test + 1 - i)

        predictions.append(yhat)

    d = avg_values[split + total_window_size + 1:]
    # d = avg_values[split + window_size :]
    rmse = sqrt(mean_squared_error(d, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions


def get_combinations(comb):
    comb = list(itertools.product(*comb))
    purged_combinations = []
    for win_size in window_size_avg_opt:
        flag_false_bit_false_trend = True
        flag_false_bit_true_trend = True
        flag_true_bit_false_trend = True
        for combination in comb:
            if win_size == combination[window_size_avg_index]:
                use_bitcoin_columns = combination[use_bitcoin_columns_index]
                use_trend_columns = combination[use_trend_columns_index]
                save = True
                purged_comb = ["", "", "", "", "", "", "", ""]
                purged_comb[window_size_avg_index] = combination[window_size_avg_index]
                purged_comb[window_size_btc_index] = combination[window_size_btc_index]
                purged_comb[window_size_trend_index] = combination[window_size_trend_index]
                purged_comb[lag_index] = combination[lag_index]

                purged_comb[use_bitcoin_columns_index] = combination[use_bitcoin_columns_index]
                purged_comb[bitcoin_columns_index] = combination[bitcoin_columns_index]

                purged_comb[use_trend_columns_index] = combination[use_trend_columns_index]
                purged_comb[trend_column_index] = combination[trend_column_index]

                if not use_bitcoin_columns and not use_trend_columns:
                    if flag_false_bit_false_trend:
                        purged_comb[window_size_btc_index] = 0
                        purged_comb[use_bitcoin_columns_index] = False
                        purged_comb[bitcoin_columns_index] = [""]
                        purged_comb[window_size_trend_index] = 0
                        purged_comb[use_trend_columns_index] = False
                        purged_comb[trend_column_index] = [""]
                        flag_false_bit_false_trend = False
                    else:
                        save = False

                if not use_bitcoin_columns and use_trend_columns:
                    if flag_false_bit_true_trend:
                        flag_false_bit_true_trend = False
                        purged_comb[window_size_btc_index] = 0
                        purged_comb[use_bitcoin_columns_index] = False
                        purged_comb[bitcoin_columns_index] = [""]
                    else:
                        save = False

                if not use_trend_columns:
                    purged_comb[window_size_trend_index] = 0
                    purged_comb[trend_column_index] = [""]

                if save:
                    purged_combinations.append(purged_comb)

    return purged_combinations


time_start = datetime.datetime.now()
print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))
seed = 5
np.random.seed(seed)

result = list()
write_file = True
plot = False

cross_validation_opt = [False]
use_bitcoin_data_opt = [True]
use_trend_column_opt = [False]
# window_size_opt = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# indow_size_opt = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14]
window_size_avg_opt = [3]
window_size_btc_opt = [3]
window_size_trend_opt = [3]

lag = [1]
bitcoin_columns_opt = btc_columns.bitcoin_columns_opt_test
trend_columns_opt = [
    ['Trend']
]

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

window_size_avg_index = 0
window_size_btc_index = 1
window_size_trend_index = 2
lag_index = 3
use_bitcoin_columns_index = 4
bitcoin_columns_index = 5
use_trend_columns_index = 6
trend_column_index = 7

combinations = [window_size_avg_opt,
                window_size_btc_opt,
                window_size_trend_opt,
                lag,
                use_bitcoin_data_opt,
                bitcoin_columns_opt,
                use_trend_column_opt,
                trend_columns_opt]
# combinations = [window_size,lag]
combinations = get_combinations(combinations)
total_models = len(combinations)
print('Quantity of Models: %s' % str(total_models))
model_count = 0

path = 'C:/tmp/bitcoin/'
# path = '/code/Project/data/'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_x_size_model_2.csv'

# To pair with the other models, this model gets 1438 first rows.
series = read_csv(path + input_file, header=0, sep=',')
series = series.iloc[::-1]

# The index is ordered in ascending mode
series.index = RangeIndex(len(series.index))

# We drop index which contains 29/02/
index_to_drop = series.loc[series['Date'] == '2016-02-29'].index.values.astype(int)[0]
series = series.drop(series.index[index_to_drop])

# The index is ordered again due to the a index was dropped
series.index = RangeIndex(len(series.index))
#print(series.head)

avg = series['Avg']
# avg_values = [10, 22, 30, 42, 50, 62, 70, 82, 90, 102]

lag_year = 365
avg_values, avg_previous = data_misc.slide_data(avg.values, lag_year)
date, date_previous = data_misc.slide_data(series['Date'].values, lag_year)
series = series.iloc[lag_year:]
# The index is ordered again due to the first rows were dropped
series.index = RangeIndex(len(series.index))

#print(series.head)
#print(avg_values[0:4])

df = DataFrame({'avg_previous': avg_previous })

#print(df.head(10))




elasticnet_model = None
lasso_model = None
knn5_model = None
knn10_model = None
sgd_model = None
lstm_model = None

for combination in combinations:
    model_count = model_count + 1
    window_size_avg = combination[window_size_avg_index]
    window_size_btc = combination[window_size_btc_index]
    window_size_trend = combination[window_size_trend_index]

    lag = combination[lag_index]
    use_bitcoin_columns = combination[use_bitcoin_columns_index]
    bitcoin_columns = combination[bitcoin_columns_index]
    use_trend_columns = combination[use_trend_columns_index]
    trend_columns = combination[trend_column_index]

    # We pair the avg_supervised column with the weight_supervised
    cut_beginning = [window_size_avg]
    if use_bitcoin_columns and use_trend_columns:
        cut_beginning = [window_size_avg, window_size_btc, window_size_trend]
    if use_bitcoin_columns and not use_trend_columns:
        cut_beginning = [window_size_avg, window_size_btc]
    if not use_bitcoin_columns and use_trend_columns:
        cut_beginning = [window_size_avg, window_size_trend]

    total_window_size = max(cut_beginning)

    print('----------')
    print(combination)
    print('Model %s / %s' % (str(model_count), str(total_models)))

    # Stationary Data
    # Difference only works for one step. Please implement for other steps
    avg_diff_values = data_misc.difference(avg_values, lag)

    # Avg values converted into supervised model
    avg_supervised = data_misc.timeseries_to_supervised(avg_diff_values, window_size_avg)
    # avg_supervised = data_misc.timeseries_to_supervised(avg_values, window_size_avg)

    supervised = avg_supervised.values

    # we cut according to the biggest window size
    supervised = supervised[total_window_size:, :]

    # For the previous year we make it supervised and diff.
    avg_previous = supervised_diff_dt(df, window_size_avg)

    # we cut according to the biggest window size
    avg_previous = avg_previous[total_window_size:, :]
    # Concatenate with numpy
    supervised = np.concatenate((avg_previous, supervised), axis=1)

    print('Window Size:         %s' % str(window_size_avg))
    print('Lag:                 %s' % str(lag))
    print('Use Bitcoin Columns: %s' % str(use_bitcoin_columns))
    if use_bitcoin_columns:
        print('Bitcoin Columns:     %s' % str(bitcoin_columns))
        df_bitcoin = DataFrame()
        for column in bitcoin_columns:
            df_bitcoin[column] = series[column]

        # bitcoin_values = df_bitcoin.values
        # bitcoin_values = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

        # btc = [110, 123, 130, 143, 150, 163, 170, 183, 190, 203]
        # trend = [212, 224, 232, 244, 252, 264, 272, 284, 292, 304]

        # df_bitcoin = DataFrame({'btc': btc,'trend': trend})
        btc_supervised = supervised_diff_dt(df_bitcoin, window_size_btc)

        # we cut according to the biggest window size
        btc_supervised = btc_supervised[total_window_size:, :]

        # Concatenate with numpy
        supervised = np.concatenate((btc_supervised, supervised), axis=1)

    print('Use Trend Columns:   %s' % str(use_trend_columns))
    if use_trend_columns:
        print('Trend Columns:     %s' % str(trend_columns))
        df_trend = DataFrame()
        for column in trend_columns:
            df_trend[column] = series[column]

        trend_supervised = supervised_diff_dt(df_trend, window_size_trend)

        # we cut according to the biggest window size
        trend_supervised = trend_supervised[total_window_size:, :]

        # Concatenate with numpy
        supervised = np.concatenate((trend_supervised, supervised), axis=1)

    # Supervised reffers either avg_supervised or the combination of avg_supervised with bitcoin_values.
    size_supervised = len(supervised)
    split = int(size_supervised * 0.80)
    train, test = supervised[0:split], supervised[split:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = data_misc.scale(train, test)
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

    if use_LASSO:
        y_predicted_es, lasso_model = algorithms.lasso(x_train, y_train, x_train, normalize=False)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        lasso_train_results.append(rmse)
        corr_lasso_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE Lasso   %.3f' % (rmse))

    # KNN5
    if use_KNN5:
        y_predicted_es, knn5_model = algorithms.knn_regressor(x_train, y_train, x_train, 5)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        knn5_train_results.append(rmse)
        corr_knn5_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE KNN(5)  %.3f' % (rmse))

    # KNN10
    if use_KNN10:
        y_hat_predicted, knn10_model = algorithms.knn_regressor(x_train, y_train, x_train, 10)
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

    if use_LSTM:
        # rmse, y_predicted = 0, 0
        nb_epoch = 1
        neurons = total_features
        print("Epoch: %i  Neurons: %i" % (nb_epoch, neurons))
        y_predicted_es, lstm_model = algorithms.lstm(x_train, y_train, x_train, batch_size=1, nb_epoch=nb_epoch,
                                                     neurons=neurons)
        rmse, y_predicted = compare_train(len_y_train, y_predicted_es)
        lstm_train_results.append(rmse)
        corr_lstm_train_results.append(data_misc.correlation(y_train, y_predicted))
        print('RMSE LSTM    %.3f' % (rmse))

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

col_win_size_avg = []
col_win_size_btc = []
col_win_size_trend = []
col_lag = []
col_use_bitcoin = []
col_bitcoin_data = []
col_use_trend = []
col_trend_data = []

for combination in combinations:
    col_win_size_avg.append(combination[window_size_avg_index])
    col_win_size_btc.append(combination[window_size_btc_index])
    col_win_size_trend.append(combination[window_size_trend_index])

    col_lag.append(combination[lag_index])
    col_use_bitcoin.append(combination[use_bitcoin_columns_index])
    col_bitcoin_data.append(''.join(str(s + ' ') for s in combination[bitcoin_columns_index]))
    col_use_trend.append(combination[use_trend_columns_index])
    col_trend_data.append(''.join(combination[trend_column_index]))

results_df['Window Size Avg'] = col_win_size_avg
results_df['Window Size BTC'] = col_win_size_btc
results_df['Window Size Trend'] = col_win_size_trend
results_df['Lag'] = col_lag
results_df['Use Bitcoin Columns'] = col_use_bitcoin
results_df['Bitcoin_Data'] = col_bitcoin_data
results_df['Use Trend Columns'] = col_use_trend
results_df['Trend Columns'] = col_trend_data

print(results_df)

if write_file:
    results_df.to_csv(path + output_file, header=True)

time_end = datetime.datetime.now()
print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
print('Duration of the script: %s' % (str(time_end - time_start)))
