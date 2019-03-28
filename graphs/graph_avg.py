import config
from matplotlib import pyplot
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import Span

import pandas as pd
import pandas as pd

from graphs import plots

input_file = config.input_file_ds
path = config.selected_path
dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
dfx = pd.read_csv(path + input_file, header=0, sep=',', nrows=1438,
                  parse_dates=['Date'],
                  date_parser=dateparse)
dfx = dfx.iloc[::-1]
title = 'Block chain Info by day'
date = dfx['Date'].values

titles = ['Open', 'High', 'Low', 'Close', 'Avg']
data = [dfx['Open'].values, dfx['High'].values, dfx['Low'].values, dfx['Close'].values
    , dfx['Avg'].values]

width = 900
height = 90
btc_color = 'blue'
usd_color = 'green'
quatity_color = 'gray'
bytes_color = 'yellow'

data = []
# data.append(dfx['Open'].values)
# data.append(dfx['High'].values)
# data.append(dfx['Low'].values)
# data.append(dfx['Close'].values)
# data.append(dfx['Avg'].values)


titles = ['Avg']

p1 = figure(x_axis_type="datetime")
p1.grid.grid_line_alpha = 0.2
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'USD'

p1.line(date, dfx['Avg'].values, color='red', legend='Price Avg Value', line_width=2)


p1.legend.location = "top_left"

show(gridplot([[p1]], plot_width=950, plot_height=400))