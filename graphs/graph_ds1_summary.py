import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt
from graphs import plots
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import config as conf
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
import pandas as pd
from bokeh.transform import factor_cmap
from graphs import util as gu



path = conf.selected_path
input_file_ds = conf.output_file_ds1

algorithms = ['elasticnet','lasso','knn', 'sgd','mlp','dummy']

target_column = 'RMSE_Val'
#target_column = 'Accu Val'

rows = []
columns = []
input_file = ""
for algorithm in algorithms:

    if algorithm == conf.algorithm_no_predcition:
        input_file = conf.output_file_no_prediction
    if algorithm == conf.algorithm_dummy:
        input_file = conf.output_file_dummy
    if algorithm == conf.algorithm_elasticnet:
        input_file = conf.output_file_elasticnet
    if algorithm == conf.algorithm_lasso:
        input_file = conf.output_file_lasso
    if algorithm == conf.algorithm_knn:
        input_file = conf.output_file_knn
    if algorithm == conf.algorithm_sgd:
        input_file = conf.output_file_sgd
    if algorithm == conf.algorithm_mlp:
        input_file = conf.output_file_mlp

    filename = conf.selected_path + input_file_ds + "_" + input_file

    # To pair with the other models, this model gets 1438 first rows.
    df = pd.read_csv(filename, sep=',')
    df['WinSize'] = 'WinSize ' + df['Window Size'].astype(str)
    df['Algorithm'] = algorithm

    columns = df.columns.values
    algorithm_rows = gu.filter_rmse(df, target_column)
    for row in algorithm_rows:
        rows.append(row)

filtered_df = pd.DataFrame(rows, columns=columns)

print(filtered_df)

algorithms = filtered_df.Algorithm.unique()
algorithms.sort()

win_sizes = filtered_df.WinSize.unique()
"""
data = [['knn', 'WinSize1', '13'], ['knn', 'WinSize2', '3'], ['knn', 'WinSize3', '4'],
        ['Lasso', 'WinSize1', '11'], ['Lasso', 'WinSize2', '4'], ['Lasso', 'WinSize3', '3'],
        ['Elastic', 'WinSize1', '10'], ['Elastic', 'WinSize2', '20'], ['Elastic', 'WinSize3', '8'],
        ['SGD', 'WinSize1', '8'], ['SGD', 'WinSize2', '3'], ['SGD', 'WinSize3', '7']]

"""
data = filtered_df[['Window Size', 'WinSize', 'Algorithm' ,'RMSE_Val','Accu Val']]
data = data.sort_values(['Window Size', 'Algorithm'], ascending=[True, True])
palette = ['#85CEB7',"#c9d9d3", "#718dbf", "#e84d60", '#F68A69','#FEE6A2']

x = [(win_size, algorithm) for win_size in win_sizes for algorithm in algorithms]
counts = sum(zip(data[target_column]), ())  # like an hstack

source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), plot_height=350,plot_width=800, title=target_column,
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
       fill_color=factor_cmap('x', palette=palette, factors=algorithms, start=1, end=2))

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None

show(p)
