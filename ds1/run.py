import ds1.data_creation as dc
import config as conf
import datetime
import ds1.knn as knn
import ds1.lasso as lasso
import ds1.dummy as dummy
import ds1.elasticnet as elasticnet
import ds1.sgd as sgd
import ds1.mlpregressor as mlpregressor
import ds1.results as results
import numpy as np


def print_results(window_size, scaler, algorithm_name, parameter_name, parameter_list,
                  x_train_val, x_val, x_test,
                  y_train_val, y_val, y_test,
                  y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list):
    rmse_tr_val_li, rmse_vl_li, rmse_te_li, y_tr_pred_li, y_val_pred_li, y_te_pred_li = results.results_overall(
        window_size, scaler, len(parameter_list),
        x_train_val, x_val, x_test,
        y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)
    # rmse, predictions = dv.compare_test(window_size, scaler, len(y_test), x_test, y_test)
    # print('RMSE %.2f'%(rmse))

    accuracy_li, ratio_up_li, ratio_down_li = results.results_accuracy_li(y_test, y_te_hat_es_list)
    y_tr_val_hat_corr_li, y_val_hat_corr_li, y_test_hat_corr_li = results.results_corr(y_train_val, y_val, y_test,
                                                                                       y_tr_val_hat_es_list,
                                                                                       y_val_hat_es_list,
                                                                                       y_te_hat_es_list)
    results.restults_df(algorithm_name, window_size, parameter_name,
                        parameter_list, rmse_tr_val_li, rmse_vl_li, rmse_te_li,
                        accuracy_li, ratio_up_li, ratio_down_li,
                        y_tr_val_hat_corr_li, y_val_hat_corr_li, y_test_hat_corr_li)


def main(use_no_prediction, use_dummy, use_elasticnet, use_lasso, use_knn, use_sgd, use_mlp):
    conf.print_conf()
    time_start = datetime.datetime.now()
    print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))

    # results.results_accuracy([100, 110, 120, 90], [101, 105, 130, 95])

    windows_sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # windows_sizes = [6]

    for window_size in windows_sizes:
        print('\n')
        supervised = dc.create_supervised_ds(window_size)
        # scaler, x_train, y_train_or, x_test, y_test = dc.split_ds_train_test(supervised)
        scaler, x_train, y_train, x_val, y_val, x_test, y_test = dc.split_ds_train_val_test(supervised)

        x_train_val = np.concatenate((x_train, x_val), axis=0)
        y_train_val = np.concatenate((y_train, y_val), axis=0)

        if use_no_prediction:
            y_tr_val_hat_es_list = [y_train_val]
            y_val_hat_es_list = [y_val]
            y_te_hat_es_list = [y_test]

            print_results(window_size, scaler, conf.algorithm_no_predcition, '-', [0],
                          x_train_val, x_val, x_test,
                          y_train_val, y_val, y_test,
                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)

        if use_dummy:
            y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = dummy.dummy(x_train, x_val, x_test)
            print_results(window_size, scaler, conf.algorithm_dummy, '-', [0],
                          x_train_val, x_val, x_test,
                          y_train_val, y_val, y_test,
                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)
        if use_elasticnet:
            alphas = np.linspace(3, 0, 50)
            y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = elasticnet.elasticnet(x_train, y_train, x_val,
                                                                                              y_val,
                                                                                              x_test)
            print_results(window_size, scaler, conf.algorithm_elasticnet, 'Alpha', alphas,
                          x_train_val, x_val, x_test,
                          y_train_val, y_val, y_test,
                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)

        if use_lasso:
            alphas = np.linspace(3, 0, 50)
            y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = lasso.lasso(x_train, y_train, x_val,
                                                                                    y_val,
                                                                                    x_test)
            print_results(window_size, scaler, conf.algorithm_lasso, 'Alpha', alphas,
                          x_train_val, x_val, x_test,
                          y_train_val, y_val, y_test,
                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)

        if use_knn:
            n_neighbors = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 120]
            y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = knn.knn(x_train, y_train, x_val,
                                                                                y_val,
                                                                                x_test)
            print_results(window_size, scaler, conf.algorithm_knn, 'Neighbors', n_neighbors,
                          x_train_val, x_val, x_test,
                          y_train_val, y_val, y_test,
                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)

        if use_sgd:
            alphas = np.linspace(5, 0, 50)
            y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = sgd.sgd(x_train, y_train, x_val,
                                                                                y_val,
                                                                                x_test)
            print_results(window_size, scaler, conf.algorithm_sgd, 'Alpha', alphas,
                          x_train_val, x_val, x_test,
                          y_train_val, y_val, y_test,
                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)

        if use_mlp:
            alphas = np.linspace(3, 0, 50)
            y_val_hat_es_list, y_tr_val_hat_es_list, y_te_hat_es_list = mlpregressor.mlpregressor(x_train, y_train,
                                                                                                  x_val,
                                                                                                  y_val,
                                                                                                  x_test)
            print_results(window_size, scaler, conf.algorithm_lstm, 'Alpha', alphas,
                          x_train_val, x_val, x_test,
                          y_train_val, y_val, y_test,
                          y_tr_val_hat_es_list, y_val_hat_es_list, y_te_hat_es_list)

    time_end = datetime.datetime.now()
    print('End time: %s' % str(time_end.strftime('%Y-%m-%d %H:%M:%S')))
    print('Duration of the script: %s' % (str(time_end - time_start)))


if __name__ == '__main__':
    use_no_prediction = False
    use_dummy = False
    use_elasticnet = False
    use_lasso = False
    use_knn = False
    use_sgd = False
    use_mlp = True
    main(use_no_prediction, use_dummy, use_elasticnet, use_lasso, use_knn, use_sgd, use_mlp)
