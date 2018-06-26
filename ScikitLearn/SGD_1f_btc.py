# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
from sklearn import linear_model
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from matplotlib import pyplot
import numpy
from sklearn.preprocessing import StandardScaler
from Util import misc
from Util import data_misc

# load dataset
series = read_csv('../Thesis/Bitcoin_historical_data_processed_1f.csv', header=0, sep='\t')

# numpy.random.seed(seed=9)

# transform data to be stationary
raw_values = series['Avg'].values
date = series['Date'].values
diff_values = data_misc.difference(raw_values, 1)

# transform data to be supervised learning
supervised = data_misc.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-365], supervised_values[-365:]

# transform the scale of the data
scaler, train_scaled, test_scaled = data_misc.scale(train, test)

X_train, y_train = train_scaled[:, 0:-1], train_scaled[:, -1]
X_test, y_test = test_scaled[:, 0:-1], test_scaled[:, -1]

error_scores = list()

clf = linear_model.SGDRegressor(max_iter=2000, verbose=True, shuffle=False)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

predictions = list()

for i in range(len(y_predicted)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = y_predicted[i]
    print("Scaled: Y_test: " + str(y) + " Yhat: " + str(yhat))

    yhat = data_misc.invert_scale(scaler, X, yhat)
    print("yhat no scaled:" + str(yhat))

    yhat = data_misc.inverse_difference(raw_values, yhat, len(test_scaled) - 0 - i)
    print("yhat no difference:" + str(yhat))
    predictions.append(yhat)

rmse = sqrt(mean_squared_error(raw_values[-365:], predictions))
print('Test RMSE: %.7f' % (rmse))

y = raw_values[-365:]

misc.print_comparison_list('Raw', raw_values[-365:], predictions)

# plot
misc.plot_line_graph2('SGD', date[-365:], raw_values[-365:], predictions)
misc.plot_data_graph2('Data', date, raw_values)




