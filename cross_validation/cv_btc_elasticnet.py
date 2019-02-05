from pandas import read_csv
from Util import data_misc
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
from Util import misc

window_size = 5  # 15
path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
series = series.iloc[::-1]

for i in range(0, 30):
    corr = series['Avg'].autocorr(lag=i)
    print('Corr: %.2f Lang: %i' % (corr, i))

avg = series['Avg']
avg_values = avg.values
# Stationary Data
diff_values = data_misc.difference(avg_values, 1)
avg_values = diff_values
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

x_train, y_train = train[:, 0:-1], train[:, -1]
x_val, y_val = val[:, 0:-1], val[:, -1]
x_test, y_test = test[:, 0:-1], test[:, -1]

print('ElasticNet - BTC')
print('Window Size %i' % (window_size))
print('Size Train %i' % (len(train)))
print('Size Val %i' % (len(val)))
print('Size Test %i' % (len(test)))

print('Size supervised %i' % (size_supervised))
print('Size raw_values %i' % (len(avg_values)))

alphas = np.linspace(3, -3, 50)
#alphas = [0.0612244898,0.0000001]
print(alphas)
print("Total Alphas")
print(len(alphas))

coefs = []
rmse_val = []
rmse_test = []

print("Train VS Val")
lasso = linear_model.ElasticNet(max_iter=1000000, normalize=True)
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(x_train, y_train)
    coefs.append(lasso.coef_)
    y_val_predicted = lasso.predict(x_val)
    rmse = sqrt(mean_squared_error(y_val, y_val_predicted))
    rmse_val.append(rmse)
    print('RMSE Elasticnet   %.3f    Alpha:  %.10f,' % (rmse, a))

print("Train + Val VS Test")
x_train_val = np.concatenate((x_train, x_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)

lasso = linear_model.ElasticNet(max_iter=1000000, normalize=True)
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(x_train_val, y_train_val)
    coefs.append(lasso.coef_)
    y_test_predicted = lasso.predict(x_test)
    rmse = sqrt(mean_squared_error(y_test, y_test_predicted))
    rmse_test.append(rmse)
    print('RMSE Elasticnet   %.3f    Alpha:  %.10f,' % (rmse, a))

rmse_avg = np.add(rmse_val, rmse_test)
rmse_avg = np.add(rmse_avg, 2)

print("Best Alpha")
best_alpha = alphas[rmse_avg.argmin()]
print(best_alpha)

print("Best RMSE of Val")
print(rmse_val[rmse_avg.argmin()])

print("Best RMSE of Test")
print(rmse_test[rmse_avg.argmin()])

print("RMSE AVG Lowest Value (Val + Test)")
print(rmse_avg.min())

print("RMSE Index from Lowest Value")
print(rmse_avg.argmin())

misc.plot_cross_validation(alphas=alphas, best_alpha=best_alpha, rmse_val=rmse_val, rmse_test=rmse_test)
