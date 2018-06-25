from sklearn.neighbors import KNeighborsRegressor
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from Util import misc
from Util import data_misc


series = read_csv('../Thesis/Bitcoin_historical_data_processed_supervised.csv', header=0, sep='\t')

# transform data to be stationary
raw_values = series['Avg'].values
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

n_neighbors = 5
neigh = KNeighborsRegressor(algorithm='kd_tree', leaf_size=30, weights='uniform', n_neighbors=n_neighbors, n_jobs=4)
neigh.fit(X_train, y_train)

predictions = list()

y_predicted = neigh.predict(X_test)
print(neigh.score(X_test,y_test))

# raw_values[-365:]

for i in range(len(y_predicted)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = y_predicted[i]
    print("Scaled: Y_test: " + str(y) + " Yhat: " + str(yhat))

    yhat = data_misc.invert_scale(scaler, X, yhat)
    print("yhat no scaled:" + str(yhat))

    yhat = data_misc.inverse_difference(raw_values, yhat, len(test_scaled) + 0 - i)
    print("yhat no difference:" + str(yhat))
    predictions.append(yhat)

rmse = sqrt(mean_squared_error(raw_values[-365:], predictions))
print('Test RMSE: %.7f' % (rmse))

y = raw_values[-365:]

#misc.print_comparison_list('RawData', y, predictions)
misc.plot_line_graph('KNN(' + str(n_neighbors) + ')', raw_values[-365:], predictions)
misc.plot_line_graph('KNN(' + str(n_neighbors) + ')', raw_values[-365:], predictions)
