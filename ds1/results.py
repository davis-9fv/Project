import ds1.data_validation as dv
import pandas as pd
import config as conf
import os.path
from sklearn.metrics import accuracy_score


def results_train_val(window_size, scaler, x_train, y_train_predicted_es):
    len_train_y = len(y_train_predicted_es)
    rmse_train, y_train_predicted = dv.compare_train(window_size, scaler, len_train_y, x_train, y_train_predicted_es)
    # print(rmse_train)
    return rmse_train, y_train_predicted


def results_val(window_size, scaler, x_val, y_val_predicted_es):
    len_val_y = len(y_val_predicted_es)
    rmse_val, y_val_predicted = dv.compare_val(window_size, scaler, len_val_y, x_val, y_val_predicted_es)
    # print(rmse_val)
    return rmse_val, y_val_predicted


def results_test(window_size, scaler, x_test, y_test_predicted_es):
    len_test_y = len(y_test_predicted_es)
    rmse_test, y_test_predicted = dv.compare_test(window_size, scaler, len_test_y, x_test, y_test_predicted_es)
    # print(rmse_test)
    return rmse_test, y_test_predicted


def results_overall(window_size, scaler, iterations, x_train, x_val, x_test, y_tr_vl_hat_es_li, y_val_hat_es_li,
                    y_te_pred_es_li):
    rmse_tr_val_li = []
    rmse_val_li = []
    rmse_test_li = []

    y_tr_val_hat_li = []
    y_val_hat_li = []
    y_test_hat_li = []

    for i in range(iterations):
        # rmse_tr_val, y_train_hat =0 , 0
        rmse_tr_val, y_train_hat = results_train_val(window_size, scaler, x_train, y_tr_vl_hat_es_li[i])
        rmse_val, y_val_hat = results_val(window_size, scaler, x_val, y_val_hat_es_li[i])
        rmse_test, y_test_hat = results_test(window_size, scaler, x_test, y_te_pred_es_li[i])

        rmse_tr_val_li.append(rmse_tr_val)
        rmse_val_li.append(rmse_val)
        rmse_test_li.append(rmse_test)

        y_tr_val_hat_li.append(y_train_hat)
        y_val_hat_li.append(y_val_hat)
        y_test_hat_li.append(y_test_hat)

    return rmse_tr_val_li, rmse_val_li, rmse_test_li, y_tr_val_hat_li, y_val_hat_li, y_test_hat_li


def restults_df(algorithm, window_size, parameter_name, parameter_list,
                rmse_tr_val_li, rmse_val_li, rmse_te_li, accu_list):
    df = pd.DataFrame({'Window Size': window_size,
                       parameter_name: parameter_list,
                       'RMSE Train + Val': rmse_tr_val_li,
                       'RMSE Val': rmse_val_li,
                       'RMSE Test': rmse_te_li,
                       'Accu': accu_list})
    output_file = ""
    if algorithm == conf.algorithm_dummy:
        output_file = conf.output_file_dummy
    if algorithm == conf.algorithm_elasticnet:
        output_file = conf.output_file_elasticnet
    if algorithm == conf.algorithm_lasso:
        output_file = conf.output_file_lasso
    if algorithm == conf.algorithm_knn:
        output_file = conf.output_file_knn
    if algorithm == conf.algorithm_sgd:
        output_file = conf.output_file_sgd
    if algorithm == conf.algorithm_lstm:
        output_file = conf.output_file_lstm

    filename = conf.selected_path + conf.output_file_extension + "_" + output_file
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', sep=',', header=False)
    else:
        df.to_csv(filename, sep=',', header=True)


def results_accuracy(y_test, y_hat):
    y_test_acc = []
    y_hat_acc = []
    for i in range(len(y_test)):
        if i < len(y_test) - 1:
            if y_test[i] > y_test[i + 1]:
                y_test_acc.append(0)
            else:
                y_test_acc.append(1)

    for i in range(len(y_test)):
        if i < len(y_test) - 1:
            if y_test[i] > y_hat[i + 1]:
                y_hat_acc.append(0)
            else:
                y_hat_acc.append(1)

    accu = accuracy_score(y_test_acc, y_hat_acc, normalize=True)
    #print(accu)
    #print(y_test_acc)
    #print(y_hat_acc)

    return accu


def results_accuracy_li(y_test_li, y_hat_li):
    accu_li = []
    for i in range(len(y_hat_li)):
        accu = results_accuracy(y_test_li, y_hat_li[i])
        accu_li.append(accu)

    return accu_li
