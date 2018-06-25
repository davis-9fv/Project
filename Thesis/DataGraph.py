# https://machinelearningmastery.com/time-series-data-stationary-python/
from matplotlib import pyplot
from pandas import read_csv
from Util import data_misc

series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')
avg_column = series["Avg"]
raw_values = series.values
diff = data_misc.difference(avg_column,1)


pyplot.plot(diff, label='BTC')
pyplot.legend()
pyplot.title('Stationary - Avg Bitcoin Prices')
pyplot.xlabel('Days')
pyplot.ylabel('USD')
pyplot.show()
