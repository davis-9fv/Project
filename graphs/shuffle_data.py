from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from graphs import plots
from util import data_misc
from sklearn.linear_model import ElasticNet
import datetime


def compare(y_test, y_predicted):
    rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse


time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))

result = list()
execute_train = False

iterations = 200
x_iteration = [x for x in range(0, iterations)]
y_rmse = [0 for x in range(0, iterations)]

for i in range(0, iterations):
    window_size = 8  # 15
    series = read_csv('C:/tmp/bitcoin/bitcoin_usd_bitcoin_block_chain_by_day.csv', header=0, sep=',')

    series = series.iloc[::-1]
    date = series['Date']
    series = series['Avg']
    date = date.iloc[window_size:]
    date = date.values

    raw_values = series.values
    raw_values = data_misc.timeseries_to_supervised(raw_values, window_size)
    #raw_values = shuffle(raw_values)
    # print(raw_values)
    raw_values = raw_values.values[window_size:, :]

    size_raw_values = len(raw_values)
    split = int(size_raw_values * 0.80)

    train, test = raw_values[0:split], raw_values[split:]
    x_train, y_train = train[:, 0:-1], train[:, -1]
    x_test, y_test = test[:, 0:-1], test[:, -1]

    print('Size raw_values %i' % (size_raw_values))

    print('------- Test --------')
    # No Prediction
    y_hat_predicted = y_test
    rmse = compare(y_test, y_hat_predicted)
    print('RMSE NoPredic  %.3f' % (rmse))

    # Dummy
    y_predicted_dummy = x_test[:, 0]
    rmse = compare(y_test, y_predicted_dummy)
    print('RMSE Dummy   %.3f' % (rmse))

    # ElasticNet
    # alphas = np.logspace(-1, 2, 70)
    # for alpha in alphas:
    # print(alpha)
    regr = ElasticNet(random_state=0, max_iter=12000, alpha=0.1)
    regr.fit(x_train, y_train)
    y_predicted_en = regr.predict(x_test)
    rmse = compare(y_test, y_predicted_en)
    print('RMSE Elastic %.3f' % (rmse))
    y_rmse[i] = rmse
    # print('Y_test')
    # print(y_test)
    print(result)

    titles = ['Y', 'ElasticNet']
    data = [y_test, y_predicted_en]

    date_test = date[split:]
    print('Length date test:' + str(len(date_test)))
    print('Length data test:' + str(len(y_test)))

# print(y)

plots.plot_one_line('Shuflle of the data', x_iteration, y_rmse, 'Iteration', 'RMSE')
print(sum(y_rmse) / float(len(y_rmse)))

time_end = datetime.datetime.now()
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Duration of the script: %s' % (str(time_end - time_start)))
