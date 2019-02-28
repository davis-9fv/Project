#!/usr/bin/env python
path_windows = 'C:/tmp/bitcoin/'
path_linux1 = '/code/Project/data/'
path_linux2 = '/home/francisco/Project/data/'
selected_path = path_windows
input_file_ds = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'

algorithm_no_predcition = 'no_prediction'
algorithm_dummy = 'dummy'
algorithm_elasticnet = 'elasticnet'
algorithm_lasso = 'lasso'
algorithm_knn = 'knn'
algorithm_sgd = 'sgd'
algorithm_lstm = 'lstm'

output_file_no_prediction = 'no_prediction.csv'
output_file_dummy = 'dummy.csv'
output_file_elasticnet = 'elasticnet.csv'
output_file_lasso = 'lasso.csv'
output_file_knn = 'knn.csv'
output_file_sgd = 'sgd.csv'
output_file_lstm = 'lstm.csv'

output_file_extension = 'ds1'

windows_os = 'Windows'
linux_os = 'Linux'
selected_os = windows_os
train_val_rate = 0.60
val_test_rate = 0.20
train_rate = 0.80
test_rate = 0.20


def print_conf():
    print('Selected OS: %s' % (selected_os))
    print('Selected Path: %s' % (selected_path))
    print('File Data Source:          %s' % (input_file_ds))
    print('Train Rate:    %.2f' % (train_rate))
    print('Test Rate:     %.2f' % (test_rate))
