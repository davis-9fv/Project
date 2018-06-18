# https://machinelearningmastery.com/time-series-data-stationary-python/

from pandas import Series
from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame
from pandas import read_csv
from sklearn.preprocessing import StandardScaler


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def inverse_difference(last_ob, value):
    return value + last_ob


series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')
raw_values = series.values

avg_column = series["Avg"]
split = int(len(avg_column) / 2)

# Check if it is stationary
x1, x2 = avg_column[0:split], avg_column[split:]
mean1, mean2 = x1.mean(), x2.mean()
var1, var2 = x1.var(), x2.var()

print('Non-stationary')
print('mean1=%.3f, mean2=%.3f' % (mean1, mean2))
print('variance1=%.3f, variance2=%.3f' % (var1, var2))
print('')
print(avg_column)
#avg_column.hist()
#pyplot.title('Non-Stationary Histogram Avg Bitcoin Prices')
# pyplot.show()


# Check the stationary data

avg_column_diff = difference(avg_column, 1)
print(avg_column_diff)
diff_x1, diff_x2 = avg_column_diff[0:split], avg_column_diff[split:]
diff_mean1, diff_mean2 = diff_x1.mean(), diff_x2.mean()
diff_var1, diff_var2 = diff_x1.var(), diff_x2.var()
print('Stationary')
print('mean1=%.3f, mean2=%.3f' % (diff_mean1, diff_mean2))
print('variance1=%.3f, variance2=%.3f' % (diff_var1, diff_var2))

avg_column_diff.hist()
pyplot.title('Stationary - Histogram Avg Bitcoin Prices')
pyplot.show()

# invert the difference
inverted = [inverse_difference(avg_column[i], avg_column_diff[i]) for i in range(len(avg_column_diff))]
print(inverted)
