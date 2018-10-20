from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import pandas as pd

series = pd.read_csv('C:/tmp/bitcoin/bitcoin_by_month.csv', header=0, sep=',')
graphs_folder = 'C:/tmp/bitcoin/graphs/'
column_name = 'weight'
column_time = 'month'

df_column = series[column_name]
df_date = series[column_time]

df_column = df_column.values
df_date = df_date.values

p1 = figure(x_axis_type="datetime", title=column_name)
p1.grid.grid_line_alpha = 0.8
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Quantity'

p1.line(pd.to_datetime(df_date), df_column, color='black', legend='Real Value')
p1.legend.location = "top_left"

output_file(graphs_folder + column_time + '_' + column_name + '.html', title="Real Value - Total of Passangers")
show(gridplot([[p1]], plot_width=950, plot_height=400))  # open a browser
