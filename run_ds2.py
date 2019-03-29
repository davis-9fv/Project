import util.data_creation as dc
import config as conf
import datetime
import algorithms.knn as knn
import algorithms.lasso as lasso
import algorithms.dummy as dummy
import algorithms.elasticnet as elasticnet
import algorithms.sgd as sgd
import algorithms.mlpregressor as mlpregressor
import results as results
import numpy as np
import btc_columns
from util import data_misc


def fill_col_list(columns, len):
    btc_col_li = []
    for i in range(len):
        btc_col_li.append(columns)

    return btc_col_li


def print_results(window_size, avg_window_size, btc_window_size,
                  scaler, algorithm_name, parameter_name, parameter_list,
                  x_train_val, x_val, x_test,
                  y_train_val, y_val, y_test,
                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                  columns, observation=""):
    rmse, y_tr_pred_li, y_val_pred_li, y_te_pred_li = results.results_overall(
        window_size, scaler, len(parameter_list),
        x_train_val, x_val, x_test,
        y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)
    # rmse, predictions = dv.compare_test(window_size, scaler, len(y_test), x_test, y_test)
    # print('RMSE %.2f'%(rmse))

    accuracy, ratio_up_li, ratio_down_li = results.accuracy_li(y_train_val, y_val, y_test,
                                                               y_tr_val_hat_es_list, y_val_hat_es_list,
                                                               y_te_hat_es_list)
    correlation = results.results_corr(y_train_val, y_val, y_test,
                                       y_tr_val_hat_es_list,
                                       y_val_hat_es_list,
                                       y_te_hat_es_list)
    results.restults_ds2_df(algorithm_name, avg_window_size, btc_window_size,
                            parameter_name,
                            parameter_list, rmse,
                            accuracy, ratio_up_li, ratio_down_li,
                            correlation,
                            columns, observation)


def main(avg_window_sizes, btc_window_sizes,
         use_no_prediction, use_dummy, use_elasticnet,
         use_lasso, use_knn, use_sgd, use_mlp):
    conf.print_conf()
    time_start = datetime.datetime.now()
    print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))

    for avg_window_size in avg_window_sizes:
        for btc_window_size in btc_window_sizes:

            for btc_col in btc_columns.bitcoin_columns_opt_pro:
                print('\n')

                series = dc.load_dataset()
                avg_supervised = dc.create_avg_supervised_ds(avg_window_size, series, False)
                btc_supervised = dc.create_btc_supervised_ds(btc_window_size, series, btc_col)

                window_size = [avg_window_size, btc_window_size]
                window_size = max(window_size)

                # we cut according to the biggest window size
                avg_supervised = avg_supervised.values[window_size:, :]
                # we cut according to the biggest window size
                btc_supervised = btc_supervised[window_size:, :]
                # Concatenate with numpy
                supervised = np.concatenate((btc_supervised, avg_supervised), axis=1)

                # scaler, x_train, y_train_or, x_test, y_test = dc.split_ds_train_test(supervised)
                scaler, x_train, y_train, x_val, y_val, x_test, y_test = dc.split_ds_train_val_test(supervised)

                x_train_val = np.concatenate((x_train, x_val), axis=0)
                y_train_val = np.concatenate((y_train, y_val), axis=0)

                if use_no_prediction:
                    y_tr_val_hat_es_list = [y_train_val]
                    y_val_hat_es_list = [y_val]
                    y_te_hat_es_list = [y_test]

                    btc_col_li = fill_col_list(btc_col, len(y_val_hat_es_list))

                    print_results(window_size, avg_window_size, btc_window_size,
                                  scaler, conf.algorithm_no_predcition, '-', [0],
                                  x_train_val, x_val, x_test,
                                  y_train_val, y_val, y_test,
                                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                                  btc_col_li)

                if use_dummy:
                    y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = dummy.dummy(x_train, x_val, x_test)

                    btc_col_li = fill_col_list(btc_col, len(y_val_hat_es_list))

                    print_results(window_size, avg_window_size, btc_window_size,
                                  scaler, conf.algorithm_dummy, '-', [0],
                                  x_train_val, x_val, x_test,
                                  y_train_val, y_val, y_test,
                                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                                  btc_col_li)
                if use_elasticnet:
                    alphas = np.linspace(2.50, 0, 40)
                    alphas = data_misc.float_presicion(alphas, 4)
                    y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = elasticnet.elasticnet(x_train, y_train,
                                                                                                      x_val,
                                                                                                      y_val,
                                                                                                      x_test,
                                                                                                      alphas)
                    btc_col_li = fill_col_list(btc_col, len(alphas))
                    print_results(window_size, avg_window_size, btc_window_size,
                                  scaler, conf.algorithm_elasticnet, 'Alpha', alphas,
                                  x_train_val, x_val, x_test,
                                  y_train_val, y_val, y_test,
                                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                                  btc_col_li)

                if use_lasso:
                    alphas = np.linspace(2.50, 0, 40)
                    alphas = data_misc.float_presicion(alphas, 4)
                    y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = lasso.lasso(x_train, y_train, x_val,
                                                                                            y_val,
                                                                                            x_test,
                                                                                            alphas)
                    btc_col_li = fill_col_list(btc_col, len(alphas))
                    print_results(window_size, avg_window_size, btc_window_size,
                                  scaler, conf.algorithm_lasso, 'Alpha', alphas,
                                  x_train_val, x_val, x_test,
                                  y_train_val, y_val, y_test,
                                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                                  btc_col_li)

                if use_knn:
                    n_neighbors = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 120]
                    y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = knn.knn(x_train, y_train, x_val,
                                                                                        y_val,
                                                                                        x_test,
                                                                                        n_neighbors)

                    btc_col_li = fill_col_list(btc_col, len(n_neighbors))
                    print_results(window_size, avg_window_size, btc_window_size,
                                  scaler, conf.algorithm_knn, 'Neighbors', n_neighbors,
                                  x_train_val, x_val, x_test,
                                  y_train_val, y_val, y_test,
                                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                                  btc_col_li)

                if use_sgd:
                    alphas = np.linspace(2.50, 0, 40)
                    alphas = data_misc.float_presicion(alphas, 4)
                    y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = sgd.sgd(x_train, y_train, x_val,
                                                                                        y_val,
                                                                                        x_test,
                                                                                        alphas)

                    btc_col_li = fill_col_list(btc_col, len(alphas))
                    print_results(window_size, avg_window_size, btc_window_size,
                                  scaler, conf.algorithm_sgd, 'Alpha', alphas,
                                  x_train_val, x_val, x_test,
                                  y_train_val, y_val, y_test,
                                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                                  btc_col_li)

                if use_mlp:
                    alphas = np.linspace(2, 0, 25)
                    alphas = data_misc.float_presicion(alphas, 4)
                    btc_col_li = fill_col_list(btc_col, len(alphas))
                    hidden_layer_1 = [5, 8, 10, 15, 20, 25, 50, 100]
                    hidden_layer_2 = [0, 5, 8, 10, 15, 20, 25, 50, 100]
                    hidden_layer_3 = [0, 5, 8, 10, 15, 20, 25, 50, 100]

                    activation = ['relu']
                    optimization = ['adam']
                    for hl1 in hidden_layer_1:
                        for hl2 in hidden_layer_2:
                            for hl3 in hidden_layer_3:
                                if not (hl2 == 0 and hl3 != 0):
                                    for act in activation:
                                        for optim in optimization:
                                            y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = mlpregressor.mlpregressor(
                                                x_train, y_train,
                                                x_val, y_val,
                                                x_test,
                                                alphas,
                                                hl1, hl2, hl3,
                                                act,
                                                optim)
                                            observation = (
                                                    "Optimization: %s, hl1: %i, hl2: %i, hl3: %i " % (
                                                optim, hl1, hl2, hl3))
                                            print_results(window_size, avg_window_size, btc_window_size,
                                                          scaler, conf.algorithm_mlp, 'Alpha', alphas,
                                                          x_train_val, x_val, x_test,
                                                          y_train_val, y_val, y_test,
                                                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list,
                                                          btc_col_li, observation)

    time_end = datetime.datetime.now()
    print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
    print('Duration of the script: %s' % (str(time_end - time_start)))


if __name__ == '__main__':
    # avg_window_sizes = [3]
    avg_window_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    btc_window_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    use_no_prediction = False
    use_dummy = False
    use_elasticnet = False
    use_lasso = False
    use_knn = False
    use_sgd = False
    use_mlp = False
    main(avg_window_sizes, btc_window_sizes,
         use_no_prediction, use_dummy, use_elasticnet,
         use_lasso, use_knn, use_sgd, use_mlp)
