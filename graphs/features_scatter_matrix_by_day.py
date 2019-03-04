import pandas as pd
from graphs import plots

from bokeh.layouts import gridplot
from bokeh.io import show

path = 'C:/tmp/bitcoin/'
#input_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
dfx = pd.read_csv(path + input_file, header=0, sep=',')
title = 'Block chain Info by day'
date = dfx['Date'].values

titles = ['Open', 'High', 'Low', 'Close', 'Avg', 'input_total_usd', 'output_total_usd']
data = [dfx['Open'].values, dfx['High'].values, dfx['Low'].values, dfx['Close'].values
    , dfx['Avg'].values, dfx['input_total_usd'].values, dfx['output_total_usd'].values]

width = 900
height = 90
btc_color = 'blue'
usd_color = 'green'
quatity_color = 'gray'
bytes_color = 'yellow'

m1 = plots.create_figure(label='Quantity', width=width, height=height, date=date,
                         column=dfx['transaction_count'].values, legend='transaction_count', color=quatity_color)
m2 = plots.create_figure(label='Quantity', width=width, height=height, date=date,
                         column=dfx['input_count'].values, legend='input_count', color=quatity_color)

m3 = plots.create_figure(label='Quantity', width=width, height=height, date=date,
                         column=dfx['output_count'].values, legend='output_count', color=quatity_color)

m4 = plots.create_figure(label='BTC', width=width, height=height, date=date,
                         column=dfx['input_total'].values, legend='input_total', color=btc_color)
m5 = plots.create_figure(label='BTC', width=width, height=height, date=date,
                         column=dfx['output_total'].values, legend='output_total', color=btc_color)

m6 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['input_total_usd'].values, legend='input_total_usd', color=usd_color)
m7 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['output_total_usd'].values, legend='output_total_usd', color=usd_color)

m8 = plots.create_figure(label='BTC', width=width, height=height, date=date,
                         column=dfx['fee_total'].values, legend='fee_total', color=btc_color)
m9 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['fee_total_usd'].values, legend='fee_total_usd', color=usd_color)

m10 = plots.create_figure(label='BTC', width=width, height=height, date=date,
                          column=dfx['generation'].values, legend='generation', color=btc_color)
m11 = plots.create_figure(label='BTC', width=width, height=height, date=date,
                          column=dfx['reward'].values, legend='reward', color=btc_color)

m12 = plots.create_figure(label='Bytes', width=width, height=height, date=date,
                          column=dfx['size'].values, legend='size', color=bytes_color)
m13 = plots.create_figure(label='Quantity', width=width, height=height, date=date,
                          column=dfx['weight'].values, legend='weight', color=quatity_color)
m14 = plots.create_figure(label='Quantity', width=width, height=height, date=date,
                          column=dfx['stripped_size'].values, legend='stripped_size', color=quatity_color)

p1 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['Open'].values, legend='Open', color=usd_color)
p2 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['High'].values, legend='High', color=usd_color)
p3 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['Low'].values, legend='Low', color=usd_color)
p4 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['Close'].values, legend='Close', color=usd_color)
p5 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['Avg'].values, legend='Avg', color=usd_color)
p6 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['input_total_usd'].values, legend='input_total_usd', color=usd_color)
p7 = plots.create_figure(label='USD', width=width, height=height, date=date,
                         column=dfx['output_total_usd'].values, legend='output_total_usd', color=usd_color)

p8 = plots.create_figure(label='Percentage', width=width, height=height, date=date,
                         column=dfx['Trend'].values, legend='Trend', color=bytes_color)


p = gridplot([m1], [m2], [m3], [m4], [m5], [m6], [m7], [m8], [m9], [m10], [m11],
             [m12], [m13], [m14], [p1], [p2], [p3], [p4], [p5], [p6], [p7], [p8])

# show the results
show(p)
