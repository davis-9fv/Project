from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from graphs import plots
import config


def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


path = config.selected_path
input_file = config.input_file_ds

dfx = pd.read_csv(path + input_file, header=0, sep=',')
print(dfx.shape)
# max_lag = dfx.shape[0]
max_lag = 700
lags = [x for x in range(0, max_lag)]
correlations = [0 for x in range(0, max_lag)]

for i in range(0, max_lag - 1):
    corr = dfx['Avg'].autocorr(lag=i)
    correlations[i] = corr * 100
    print('Corr: %.2f Lang: %i' % (corr, i))

plots.plot_one_line('Correlation by day', lags, correlations, 'Day (Lag)', 'Correlation')
