# https://machinelearningmastery.com/time-series-data-stationary-python/
from matplotlib import pyplot
from pandas import read_csv
from Util import data_misc
from Util import misc
from Util import data_misc
import pandas as pd

series = read_csv('../Thesis/Bitcoin_historical_data_processed_1f.csv', header=0, sep='\t')
# transform data to be stationary
raw_values = series['Avg'].values
date = series['Date'].values
diff = data_misc.difference(raw_values, 1)

rolmean = pd.rolling_mean(raw_values, window=15)
rolstd = pd.rolling_std(raw_values, window=15)

titles = ['Raw Data', 'Rolling Mean', 'Standard Mean']
data = [raw_values, rolmean, rolstd]
misc.plot_lines_graph('Data ', date, titles, data)
misc.plot_data_graph2('Stationary ', date, diff)
