import pandas as pd
import config as conf
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
import pandas as pd
from bokeh.transform import factor_cmap
from graphs import util as gu

path = conf.selected_path
input_file_ds = conf.output_file_ds1

algorithms = ['elasticnet', 'lasso', 'knn', 'sgd', 'mlp', 'dummy']

target_column = 'RMSE_Val'
# target_column = 'Accu Val'

rows = []
columns = []
input_file = ""
new_columns = ['WinSize', 'Window Size', 'Algorithm', 'RMSE']
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
    algorithm_df = pd.DataFrame(algorithm_rows, columns=columns)



    for index, row in algorithm_df.iterrows():
        row_val = [row['WinSize'], row['Window Size'], row['Algorithm'] + ' val', row['RMSE_Val']]
        row_test = [row['WinSize'], row['Window Size'], row['Algorithm'] + ' test', row['RMSE Test']]
        rows.append(row_val)
        rows.append(row_test)


filtered_df = pd.DataFrame(rows, columns=new_columns)

print(filtered_df)

algorithms = filtered_df.Algorithm.unique()
algorithms.sort()

win_sizes = filtered_df.WinSize.unique()

#data = filtered_df[['Window Size', 'WinSize', 'Algorithm', 'RMSE_Val', 'Accu Val']]
data = filtered_df.sort_values(['Window Size', 'Algorithm'], ascending=[True, True])
palette = ['#85CEB7','#85CEB7', "#c9d9d3", "#c9d9d3", "#718dbf", "#718dbf",
           "#e84d60", "#e84d60", '#F68A69', '#F68A69', '#FEE6A2', '#FEE6A2']

x = [(win_size, algorithm) for win_size in win_sizes for algorithm in algorithms]
counts = sum(zip(data['RMSE']), ())  # like an hstack

source = ColumnDataSource(data=dict(x=x, counts=counts))

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
p = figure(x_range=FactorRange(*x), plot_height=350, plot_width=1350, title='RMSE',
            tools=TOOLS)


p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
       fill_color=factor_cmap('x', palette=palette, factors=algorithms, start=1, end=2))

p.y_range.start = 0
p.x_range.range_padding = 0.02
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None


show(p)
