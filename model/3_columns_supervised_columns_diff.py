from sklearn.utils import shuffle
import datetime
from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc
import numpy as np
import itertools
from model import values


def supervised_diff_dt(df_data, window_size):
    processed_columns = []
    for column_name in df_data:
        values_column = df_data[column_name]
        diff_col = data_misc.difference(values_column, 1)
        supervised_col = data_misc.timeseries_to_supervised(diff_col, window_size)
        # We drop the last column from weight_supervised because it is not the target we want
        supervised_col = supervised_col.values[:, :-1]
        processed_columns.append(supervised_col)

    flag = True
    result = []
    for col in processed_columns:
        if flag:
            result = col
            flag = False
        else:
            result = np.concatenate((result, col), axis=1)

    return result


avg_window_size = 3
btc_window_size = 4
avg = [10, 22, 30, 42, 50, 62, 70, 82, 90, 102]
btc = [110, 123, 130, 143, 150, 163, 170, 183, 190, 203]
trend = [212, 224, 232, 244, 252, 264, 272, 284, 292, 304]

df = DataFrame({'btc': btc,
                'trend': trend})
print(btc)

avg_diff = data_misc.difference(avg, 1)
avg_supervised = data_misc.timeseries_to_supervised(avg_diff, avg_window_size)
print(avg_supervised)

btc_supervised = supervised_diff_dt(df, btc_window_size)

# We pair the avg_supervised column with the weight_supervised
cut_beginning = [avg_window_size, btc_window_size]
cut_beginning = max(cut_beginning)

avg_supervised = avg_supervised.values[cut_beginning:, :]
btc_supervised = btc_supervised[cut_beginning:, :]

# Concatenate with numpy
supervised = np.concatenate((btc_supervised, avg_supervised), axis=1)

print(supervised)
print("No diff")
for i in range(len(avg_diff)):
    avg_un_diff = data_misc.inverse_difference(avg, avg_diff[i], len(avg_diff) + 1 - i)
    print(avg_un_diff)
print("End")
