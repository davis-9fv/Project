from sklearn.utils import shuffle
import datetime
from Util import algorithm
from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
import numpy as np
import itertools
from model import values


def compare(y_test, y_predicted):
    predictions = list()
    for i in range(len(y_test)):
        X = x_test[i]
        yhat = y_predicted[i]
        yhat = data_misc.invert_scale(scaler, X, yhat)

        # Stationary
        d = avg_values[split + window_size - 1:]
        yhat = data_misc.inverse_difference(d, yhat, len(y_test) + 1 - i)

        predictions.append(yhat)

    d = avg_values[split + window_size + 1:]
    # d = avg_values[split + window_size :]
    rmse = sqrt(mean_squared_error(d, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions


def get_combinations(comb):
    comb = list(itertools.product(*comb))
    purged_combinations = []
    for win_size in window_size_opt:
        flag_false_bit_false_trend = True
        flag_false_bit_true_trend = True
        flag_true_bit_false_trend = True
        for combination in comb:
            if win_size == combination[window_size_index]:
                use_bitcoin_columns = combination[use_bitcoin_columns_index]
                use_trend_columns = combination[use_trend_columns_index]
                save = True
                purged_comb = ["", "", "", "", "", ""]
                purged_comb[window_size_index] = combination[window_size_index]
                purged_comb[lag_index] = combination[lag_index]

                purged_comb[use_bitcoin_columns_index] = combination[use_bitcoin_columns_index]
                purged_comb[bitcoin_columns_index] = combination[bitcoin_columns_index]

                purged_comb[use_trend_columns_index] = combination[use_trend_columns_index]
                purged_comb[trend_column_index] = combination[trend_column_index]

                if not use_bitcoin_columns and not use_trend_columns:
                    if flag_false_bit_false_trend:
                        purged_comb[use_bitcoin_columns_index] = False
                        purged_comb[bitcoin_columns_index] = [""]
                        purged_comb[use_trend_columns_index] = False
                        purged_comb[trend_column_index] = [""]
                        flag_false_bit_false_trend = False
                    else:
                        save = False

                if not use_bitcoin_columns and use_trend_columns:
                    if flag_false_bit_true_trend:
                        flag_false_bit_true_trend = False
                        purged_comb[use_bitcoin_columns_index] = False
                        purged_comb[bitcoin_columns_index] = [""]
                    else:
                        save = False

                if not use_trend_columns:
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
use_trend_column_opt = [True]
# window_size_opt = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# window_size_opt = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
window_size_opt = [5]

lag = [1]
bitcoin_columns_opt = values.bitcoin_columns_opt_all
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

normal_results = []
dummy_results = []
elastic_results = []
lasso_results = []
knn5_results = []
knn10_results = []
sgd_results = []
lstm_results = []
corr_normal_results = []
corr_dummy_results = []
corr_elastic_results = []
corr_lasso_results = []
corr_knn5_results = []
corr_knn10_results = []
corr_sgd_results = []
corr_lstm_results = []

window_size_index = 0
lag_index = 1
use_bitcoin_columns_index = 2
bitcoin_columns_index = 3
use_trend_columns_index = 4
trend_column_index = 5

combinations = [window_size_opt,
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
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'main_window_x_size_results2.csv'

# To pair with the other models, this model gets 1438 first rows.
series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
series = series.iloc[::-1]
avg = series['Avg']
avg_values = avg.values

for combination in combinations:
    model_count = model_count + 1
    window_size = combination[window_size_index]
    lag = combination[lag_index]
    use_bitcoin_columns = combination[use_bitcoin_columns_index]
    bitcoin_columns = combination[bitcoin_columns_index]
    use_trend_columns = combination[use_trend_columns_index]
    trend_columns = combination[trend_column_index]

    print('----------')
    print(combination)
    print('Model %s / %s' % (str(model_count), str(total_models)))

    # Stationary Data
    # Difference only works for one step. Please implement for other steps
    avg_diff_values = data_misc.difference(avg_values, lag)
    # Avg values converted into supervised model
    avg_supervised = data_misc.timeseries_to_supervised(avg_diff_values, window_size)
    # The first [Window size number] contains zeros which need to be cut.
    avg_supervised = avg_supervised.values[window_size:, :]
    supervised = avg_supervised
    print('Window Size:         %s' % str(window_size))
    print('Lag:                 %s' % str(lag))
    print('Use Bitcoin Columns: %s' % str(use_bitcoin_columns))
    if use_bitcoin_columns:
        print('Bitcoin Columns:     %s' % str(bitcoin_columns))
        df_bitcoin = DataFrame()
        for column in bitcoin_columns:
            df_bitcoin[column] = series[column]

        bitcoin_values = df_bitcoin.values
        # We cut the first or last element to pair with the diff values.
        # bitcoin_values = bitcoin_values[1:, :]
        bitcoin_values = bitcoin_values[:-1, :]
        # We cut from the beginning to pair with the supervised values.
        bitcoin_values = bitcoin_values[:-window_size, :]
        # Concatenate with numpy
        supervised = np.concatenate((bitcoin_values, supervised), axis=1)

    print('Use Trend Columns:   %s' % str(use_trend_columns))
    if use_trend_columns:
        print('Trend Columns:     %s' % str(trend_columns))
        df_trend = DataFrame()
        for column in trend_columns:
            df_trend[column] = series[column]

        trend_values = df_trend.values
        # We cut the first or last element to pair with the diff values.
        # bitcoin_values = bitcoin_values[1:, :]
        trend_values = trend_values[:-1, :]
        # We cut from the beginning to pair with the supervised values.
        trend_values = trend_values[:-window_size, :]
        # Concatenate with numpy
        supervised = np.concatenate((trend_values, supervised), axis=1)

    # Supervised reffers either avg_supervised or the combination of avg_supervised with bitcoin_values.
    size_supervised = len(supervised)
    split = int(size_supervised * 0.80)
    train, test = supervised[0:split], supervised[split:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = data_misc.scale(train, test)
    x_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
    x_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

    print('Total of supervised data: %i' % (size_supervised))

    # Use of the algorithms
    # No Prediction
    y_hat_predicted_es = y_test
    rmse, y_hat_predicted = compare(y_test, y_hat_predicted_es)
    normal_results.append(rmse)
    corr_normal_results.append(data_misc.correlation(y_test, y_hat_predicted))
    print('RMSE NoPredic  %.3f' % (rmse))

    if use_Dummy:
        y_predicted_dummy_es = x_test[:, -1]
        rmse, y_predicted_dummy = compare(y_test, y_predicted_dummy_es)
        dummy_results.append(rmse)
        corr_dummy_results.append(data_misc.correlation(y_test, y_predicted_dummy))
        print('RMSE Dummy   %.3f' % (rmse))

    if use_ElasticNet:
        y_predicted_en_es, y_future_en_es = algorithm.elastic_net(x_train, y_train, x_test, y_test, normalize=False)
        rmse, y_predicted_en = compare(y_test, y_predicted_en_es)
        elastic_results.append(rmse)
        corr_elastic_results.append(data_misc.correlation(y_test, y_predicted_en))
        print('RMSE Elastic %.3f' % (rmse))

    if use_LASSO:
        y_predicted_la_sc = algorithm.lasso(x_train, y_train, x_test, normalize=False)
        rmse, y_predicted_la = compare(y_test, y_predicted_la_sc)
        lasso_results.append(rmse)
        corr_lasso_results.append(data_misc.correlation(y_test, y_predicted_la))
        print('RMSE Lasso   %.3f' % (rmse))

    if use_KNN5:
        y_predicted_knn5_es = algorithm.knn_regressor(x_train, y_train, x_test, 5)
        rmse, y_predicted_knn5 = compare(y_test, y_predicted_knn5_es)
        knn5_results.append(rmse)
        corr_knn5_results.append(data_misc.correlation(y_test, y_predicted_knn5))
        print('RMSE KNN(5)  %.3f' % (rmse))

    if use_KNN10:
        y_predicted_knn10_es = algorithm.knn_regressor(x_train, y_train, x_test, 10)
        rmse, y_predicted_knn10 = compare(y_test, y_predicted_knn10_es)
        knn10_results.append(rmse)
        corr_knn10_results.append(data_misc.correlation(y_test, y_predicted_knn10))
        print('RMSE KNN(10) %.3f' % (rmse))

    if use_SGD:
        y_predicted_sgd_es = algorithm.sgd_regressor(x_train, y_train, x_test)
        rmse, y_predicted_sgd = compare(y_test, y_predicted_sgd_es)
        sgd_results.append(rmse)
        corr_sgd_results.append(data_misc.correlation(y_test, y_predicted_sgd))
        print('RMSE SGD     %.3f' % (rmse))

    if use_LSTM:
        rmse, y_predicted_lstm = 0, 0
        lstm_results.append(rmse)
        corr_lstm_results.append(data_misc.correlation(y_test, y_test))
        print('RMSE LSTM    %.3f' % (rmse))

    print('----------')

# Get results
results_df = DataFrame({'Normal': normal_results,
                        'Normal Corr': corr_normal_results,
                        'Dummy': dummy_results,
                        'Dummy Corr': corr_dummy_results,
                        'ElasticNet': elastic_results,
                        'ElasticNet Corr': corr_elastic_results,
                        'Lasso': lasso_results,
                        'Lasso Corr': corr_lasso_results,
                        'KNN5': knn5_results,
                        'KNN5 Corr': corr_knn5_results,
                        'KNN10': knn10_results,
                        'KNN10 Corr': corr_knn10_results,
                        'SGD': sgd_results,
                        'SGD Corr': corr_sgd_results,
                        'LSTM': lstm_results,
                        'LSTM Corr': corr_lstm_results})

col_win_size = []
col_lag = []
col_use_bitcoin = []
col_bitcoin_data = []
col_use_trend = []
col_trend_data = []

for combination in combinations:
    col_win_size.append(combination[window_size_index])
    col_lag.append(combination[lag_index])
    col_use_bitcoin.append(combination[use_bitcoin_columns_index])
    col_bitcoin_data.append(''.join(str(s + ' ') for s in combination[bitcoin_columns_index]))
    col_use_trend.append(combination[use_trend_columns_index])
    col_trend_data.append(''.join(combination[trend_column_index]))

results_df['Window Size'] = col_win_size
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
