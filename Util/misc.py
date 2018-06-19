from matplotlib import pyplot


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
