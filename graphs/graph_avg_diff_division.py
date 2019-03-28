import pandas as pd
from util import data_misc
from sklearn.preprocessing import StandardScaler
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
series = pd.read_csv(path + input_file, header=0, sep=',', nrows=1438,
                  parse_dates=['Date'],
                  date_parser=dateparse)
# series = read_csv(path + input_file, header=0, sep=',', nrows=8)

series = series.iloc[::-1]

avg = series['Avg']
date = series['Date'].values[1:]
avg_values = avg.values
# print(avg)

# Stationary Data
diff_values = data_misc.difference(avg_values, 1)

diff_values = diff_values.values
diff_values = diff_values.reshape(diff_values.shape[0], 1)
#scaler = StandardScaler()
#scaler = scaler.fit(diff_values)
#scaled_values = scaler.transform(diff_values)

print("Diff values")
x = []
for i in range(len(diff_values)):
    x.append(diff_values[i][0])
    print(diff_values[i][0])

print("unDiff values")

index = []
for i in range(len(diff_values)):
    index.append(i)
    yhat = diff_values[i]
    yhat = data_misc.inverse_difference(avg_values, yhat, len(diff_values) + 1 - i)
    # print(yhat)

p1 = figure(x_axis_type="datetime")
p1.grid.grid_line_alpha = 0.2
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'USD'

p1.line(date, x, color="#718dbf", legend='Avg Diff', line_width=2)

p1.legend.location = "top_left"

show(gridplot([[p1]], plot_width=950, plot_height=400))
