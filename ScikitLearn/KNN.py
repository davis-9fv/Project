from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


series = read_csv('../tests/shampoo-sales2.csv', header=0)
raw_values = series.values
train, test = raw_values[0:40], raw_values[40:50]

xTrain, yTrain = train[:, 0], train[:, 1]
xTrain = xTrain.reshape(xTrain.shape[0], 1)

neigh = KNeighborsRegressor(n_neighbors=6)
neigh.fit(xTrain, yTrain)

xTest, yTest = test[:, 0], test[:, 1]
xTest = xTest.reshape(xTest.shape[0], 1)
print(neigh.predict(xTest))

for i in range(len(test)):
    # make one-step forecast
    xTest, yTest = test[i, 0], test[i, 1]

    neigh.fit(xTrain, yTrain)
    yhat = neigh.predict(xTest)
    print("xTest:" + str(xTest) + " yHat:" + str(yhat) + " yTest:" + str(yTest))
