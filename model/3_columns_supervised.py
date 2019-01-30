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

avg_window_size = 4
weight_window_size = 2
avg = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
btc = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
trend = [210, 220, 230, 240, 250, 260, 270, 280, 290, 300]

#btc=[[110,210], [120,220], [130,230], [140,240], [150,250], [160,260], [170,270], [180,280], [190,290], [200,300]]
print(btc)

avg_supervised = data_misc.timeseries_to_supervised(avg, avg_window_size)
btc_supervised = data_misc.timeseries_to_supervised(btc, weight_window_size)

# We drop the last column from weight_supervised because it is not the target we want
btc_supervised = btc_supervised.values[:,:-1]

# We pair the avg_supervised column with the weight_supervised
cut_beginning = [avg_window_size, weight_window_size]
cut_beginning = max(cut_beginning)

avg_supervised = avg_supervised.values[cut_beginning:, :]
btc_supervised = btc_supervised[cut_beginning:, :]

# Concatenate with numpy
supervised = np.concatenate((btc_supervised, avg_supervised), axis=1)

print(supervised)
print("End")
