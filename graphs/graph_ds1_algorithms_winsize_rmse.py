import config as conf
import pandas as pd
from graphs import util as gu
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import time
import numpy as np
from bokeh.models import Legend
from bokeh.models import LinearAxis, Range1d

"""
For each model plot the following:
the color of the line represents the algorithm
the x represents the window size
the y represents the RMSE

"""

path = conf.selected_path
input_file_ds = conf.output_file_ds1

algorithms = ['elasticnet', 'lasso', 'knn', 'sgd', 'mlp']
algorithms = ['elasticnet', 'lasso', 'knn', 'sgd', 'mlp']

# algorithms = ['knn']

filter_column = 'RMSE_Val'

# y_column_val = 'Accu Test'
# y_column_test = 'Accu_Val'

selected_range = None
y_range = None
y_column_val = None
y_column_test = None
y_label = None
use_rmse = True
if use_rmse:
    y_label = 'RMSE'
    rmse_range = (220, 500)

    y_range = rmse_range
    y_column_val = 'RMSE_Val'
    #y_column_test = 'RMSE Test'
    y_column_test = 'RMSE Train + Val'
else:
    y_label = 'Accuracy'
    accu_range = (0, 1)
    y_range = accu_range
    y_column_val = 'Accu Val'
    y_column_test = 'Accu Test'
    y_column_test = 'Accu Train + Val'

columns = []
input_file = ""

i = 0
colors = ['skyblue', 'blue', 'black', 'gray', 'green', 'yellow', 'pink', 'orange', 'maroon', 'red']
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
p1 = figure(y_range=selected_range, tools=TOOLS)

p1.grid.grid_line_alpha = 0.2
p1.xaxis.axis_label = 'Window Size'
p1.yaxis.axis_label = y_label

for algorithm in algorithms:

    x_column = ''
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
    rows_best_rmse_of_win_size = gu.filter_rmse(df, filter_column)
    algorithm_df = pd.DataFrame(rows_best_rmse_of_win_size, columns=columns)

    p1.line(algorithm_df['Window Size'].values, algorithm_df[y_column_val].values,
            color=colors[i], legend='Val ' + algorithm,
            line_width=1.5)
    p1.circle(algorithm_df['Window Size'].values, algorithm_df[y_column_val].values,
              color=colors[i], size=3)

    p1.line(algorithm_df['Window Size'].values, algorithm_df[y_column_test].values,
            color=colors[i], legend='Train ' + algorithm,
            line_dash='12 7', line_width=1.5)
    p1.circle(algorithm_df['Window Size'].values, algorithm_df[y_column_test].values,
              color=colors[i], size=3)

    p1.legend.label_text_font_size = '9.5pt'
    p1.legend.location = "top_right"

    i = i + 1

show(gridplot([[p1]], plot_width=750, plot_height=400))
# break
