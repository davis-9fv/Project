import numpy as np
import pandas as pd


from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT


from bokeh.plotting import figure, show, output_file
from bokeh.palettes import brewer

N = 20
cats = 10

df = pd.read_csv('Bitcoin-Historical-Data_28april2013-23april2018.csv', sep='\t')
gammas = df[['Date', 'Low']]

#gammas.to_csv('myDataFrame.csv')
print(gammas.ix[:,'Low'] )
print(gammas.ix[:,'Date'] )
#print(gammas['Low'])
#print(pd.to_datetime(gammas['Date']))

p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
p1.grid.grid_line_alpha = 0.8
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Low'

p1.line(pd.to_datetime(gammas['Date']), gammas['Low'], color='#000000', legend='Bitcoin')
p1.legend.location = "top_left"

#aapl = np.array(gammas['Low'])
#aapl_dates = np.array(pd.to_datetime(gammas['Date']))

window_size = 90
window = np.ones(window_size) / float(window_size)
#aapl_avg = np.convolve(aapl, window, 'same')

output_file("stocks.html", title="stocks.py example")

show(gridplot([[p1]], plot_width=600, plot_height=400))  # open a browser
