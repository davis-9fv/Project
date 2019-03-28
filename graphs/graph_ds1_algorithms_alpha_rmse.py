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
the color of the line represents the windo size
the x represents the alpha
the y represents the RMSE
the the other y axe draw the accuracy. 

"""

path = conf.selected_path
input_file_ds = conf.output_file_ds1

algorithms = ['elasticnet', 'lasso', 'knn', 'sgd', 'mlp']
#algorithms = ['mlp']

target_column = 'RMSE_Val'
# target_column = 'Accu Val'


columns = []
input_file = ""
for algorithm in algorithms:

    x_column = ''
    if algorithm == conf.algorithm_no_predcition:
        input_file = conf.output_file_no_prediction
    if algorithm == conf.algorithm_dummy:
        input_file = conf.output_file_dummy
    if algorithm == conf.algorithm_elasticnet:
        input_file = conf.output_file_elasticnet
        x_column = 'Alpha'
    if algorithm == conf.algorithm_lasso:
        input_file = conf.output_file_lasso
        x_column = 'Alpha'
    if algorithm == conf.algorithm_knn:
        input_file = conf.output_file_knn
        x_column = 'Neighbors'
    if algorithm == conf.algorithm_sgd:
        input_file = conf.output_file_sgd
        x_column = 'Alpha'
    if algorithm == conf.algorithm_mlp:
        input_file = conf.output_file_mlp
        x_column = 'Alpha'

    filename = conf.selected_path + input_file_ds + "_" + input_file

    # To pair with the other models, this model gets 1438 first rows.
    df = pd.read_csv(filename, sep=',')
    df['WinSize'] = 'WinSize ' + df['Window Size'].astype(str)
    df['Algorithm'] = algorithm

    columns = df.columns.values
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    p1 = figure(y_range=(220, 500), tools=TOOLS,
                )

    p1.grid.grid_line_alpha = 0.2
    p1.xaxis.axis_label = x_column
    p1.yaxis.axis_label = 'RMSE'

    # Second y axis
    p1.extra_y_ranges = {"foo": Range1d(start=0, end=1)}
    p1.add_layout(LinearAxis(y_range_name="foo"), 'right')

    colors = ['', '', '', 'skyblue', 'blue', 'black','gray', 'green',  'yellow', 'pink', 'orange', 'maroon', 'red']

    i = 0
    for win_size in range(8, 13):
        algorithm_df = None
        color = colors[win_size]
        if algorithm != conf.algorithm_mlp:
            algorithm_df = gu.get_rows(df, win_size, target_column)
        else:
            alphas = df.Alpha.unique()
            rows = []
            for alpha in alphas:
                row = gu.get_first_row_alpha(df, win_size, target_column, alpha)
                rows.append(row)
            algorithm_df = pd.DataFrame(rows, columns=columns)

        algorithm_df = algorithm_df.sort_values(by=[x_column], ascending=True)
        print(algorithm_df[x_column].values)

        # p1.y_range = (200,500)
        p1.line(algorithm_df[x_column].values, algorithm_df['RMSE_Val'].values,
                color=color, legend='Val ' + algorithm + ' WinSize:' + str(win_size),
                line_width=1.5)
        p1.line(algorithm_df[x_column].values, algorithm_df['RMSE Test'].values,
                color=color, legend='Test ' + algorithm + ' WinSize:' + str(win_size),
                line_dash='12 7', line_width=1.5)

        p1.line(algorithm_df[x_column].values, algorithm_df['Accu Val'].values,
                color=color, legend='Val Accu ' + algorithm + ' WinSize:' + str(win_size),
                line_dash='2 5', line_width=2,
                y_range_name="foo")

        i = i + 1
        p1.legend.label_text_font_size = '9.5pt'

    p1.legend.location = "top_right"
    show(gridplot([[p1]], plot_width=950, plot_height=500))
    time.sleep(5)

    # break
