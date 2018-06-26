from matplotlib import pyplot
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

import pandas as pd


def print_comparison(title, expected, prediction):
    print('%s:: Y expected: %.3f   Y predicted: %.3f' % (title, expected, predictions))


def print_comparison_list(title, expected, predictions):
    for i in range(len(predictions)):
        print('%s:: Y expected: %.3f   Y predicted: %.3f' % (title, expected[i], predictions[i]))


def plot_line_graph(algorithm_name="Unkown", expected=[], predictions=[]):
    pyplot.plot(expected, label='Real Value')
    pyplot.plot(predictions, label='Predicted Value', color='red')
    pyplot.legend()
    pyplot.title(algorithm_name + ' - Avg Bitcoin Prices')
    pyplot.xlabel('Days')
    pyplot.ylabel('USD')
    pyplot.show()


def plot_data_graph(algorithm_name="Unkown", data=[]):
    pyplot.plot(data, label='Real Value')
    pyplot.legend()
    pyplot.title(algorithm_name + ' - Avg Bitcoin Prices')
    pyplot.xlabel('Days')
    pyplot.ylabel('USD')
    pyplot.show()


def plot_line_graph2(algorithm_name="Unkown", date=[], expected=[], predictions=[]):
    p1 = figure(x_axis_type="datetime", title=algorithm_name + ' - Avg Bitcoin Prices')
    p1.grid.grid_line_alpha = 0.8
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'USD'

    p1.line(pd.to_datetime(date), predictions, color='blue', legend='Predicted Value', line_dash="2 2")
    p1.line(pd.to_datetime(date), expected, color='red', legend='Expected Value', line_dash="4 4")
    p1.legend.location = "top_left"

    output_file('../tmp/' + algorithm_name + '.html', title="Avg Bitcoin Prices")
    show(gridplot([[p1]], plot_width=950, plot_height=400))  # open a browser


def plot_data_graph2(algorithm_name="Unkown", date=[], data=[]):
    p1 = figure(x_axis_type="datetime", title=algorithm_name + ' - Avg Bitcoin Prices')
    p1.grid.grid_line_alpha = 0.8
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'USD'

    p1.line(pd.to_datetime(date), data, color='black', legend='Real Value')
    p1.legend.location = "top_left"

    output_file("../tmp/RealValue.html", title="Real Value - Avg Bitcoin Prices")
    show(gridplot([[p1]], plot_width=950, plot_height=400))  # open a browser
