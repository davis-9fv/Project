# https://machinelearningmastery.com/time-series-data-stationary-python/
from matplotlib import pyplot
from pandas import read_csv

series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')
raw_values = series.values

avg_column = series["Avg"]
pyplot.plot(avg_column, label='BTC')
pyplot.legend()
pyplot.title('Avg Bitcoin Prices')
pyplot.xlabel('Days')
pyplot.ylabel('USD')
pyplot.show()
