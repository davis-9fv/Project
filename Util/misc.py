from matplotlib import pyplot
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

import pandas as pd

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

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
    p1 = figure(x_axis_type="datetime", title=algorithm_name + ' - Total of Passangers')
    p1.grid.grid_line_alpha = 0.8
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Quantity'

    p1.line(pd.to_datetime(date), data, color='black', legend='Real Value')
    p1.legend.location = "top_left"

    output_file("../tmp/RealValue.html", title="Real Value - Total of Passangers")
    show(gridplot([[p1]], plot_width=950, plot_height=400))  # open a browser


def plot_lines_graph(title="Unkown", date=[], titles=[], data=[]):
    p1 = figure(x_axis_type="datetime", title=title + ' - Total of Passangers')
    p1.grid.grid_line_alpha = 1
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Quantity'
    colors = ['red', 'blue', 'black', 'yellow', 'green', 'gray', 'pink', 'orange']
    line_dash = ['4 3', '5 4', '6 5', '7 6', '8 7', '9 8', '10 9', '10 9' ]

    # p1.line(pd.to_datetime(date), [0,1000], line_width=2)

    for i in range(len(titles)):
        p1.line(pd.to_datetime(date), data[i], color=colors[i], legend=titles[i], line_dash=line_dash[i], line_width=1)

    p1.legend.location = "top_left"

    output_file('../tmp/' + title + '.html', title=title)
    show(gridplot([[p1]], plot_width=950, plot_height=400))  # open a browser


def plot_one_line(title="Unkown", x=[], y=[], x_title='x', y_title='y', legend=''):
    p1 = figure(title=title, tools=TOOLS, tooltips=[(x_title, '@x'), (y_title, '@y')])
    p1.grid.grid_line_alpha = 0.8
    p1.xaxis.axis_label = x_title
    p1.yaxis.axis_label = y_title
    p1.line(x, y, color='blue', legend=legend)
    p1.legend.location = "top_left"
    output_file('../tmp/' + title + '.html', title=title)
    show(gridplot([[p1]], plot_width=700, plot_height=300))  # open a browser


def create_figure(label='label', width=800, height=150, date=[], column=[], legend='legend', color='blue'):
    p = figure(width=width, plot_height=height)
    p.line(pd.to_datetime(date), column, legend=legend, color=color)
    p.yaxis.axis_label = label
    p.legend.location = "top_left"
    return p
