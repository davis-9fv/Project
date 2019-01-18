from Util import algorithm
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
from sklearn.utils import shuffle


def compare_test(y_test, y_predicted):
    predictions = list()
    for i in range(len(y_test)):
        yhat = y_predicted[i]
        d = raw_values
        yhat = data_misc.inverse_difference(d, yhat, len(y_test) + 1 - i)
        predictions.append(yhat)

    d = raw_values[split + 1:]
    rmse = sqrt(mean_squared_error(d, predictions))
    # rmse = sqrt(mean_squared_error(y_test, y_predicted))
    return rmse, predictions


shuffle_data = True
raw_values = [10.1, 20.2, 10.3, 40.4, 50.5, 30.6, 0.7, 100.8, 20.9, 10.10, 40.11, 50.12, 30.13]
size_raw_values = len(raw_values)
split = int(size_raw_values * 0.80)

# Stationary Data
diff_values = data_misc.difference(raw_values, 1)
print(diff_values)
diff_values = diff_values.values
if shuffle_data:
    suffled_values = shuffle(diff_values)
    print('Shuffled_Values')
    print(suffled_values)
    index_values = [x for x in range(0, len(diff_values))]
    for i in range(0, len(diff_values)):
        for j in range(0, len(suffled_values)):
            if diff_values[i] == suffled_values[j]:
                index_values[j] = i
                break

    print('Indexes')
    print(index_values)
    for i in range(0, len(suffled_values)):
        for j in range(0, len(suffled_values)):
            if i == index_values[j]:
                print(suffled_values[j])


train, test = diff_values[0:split], diff_values[split:]

rmse, y_predicted = compare_test(test, test)
print('RMSE NoPredic  %.3f' % (rmse))
print(y_predicted)
