from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from Util import data_misc

series = read_csv('../Thesis/Bitcoin_historical_data_processed.csv', header=0, sep='\t')
X = series["Avg"]
X = X.values
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# p is the number of autoregressive terms,
# d is the number of nonseasonal differences needed for stationarity, and
# q is the number of lagged forecast errors in the prediction equation.

p = 1
d = 1
q = 0
print('P: %.2f, D: %.2f, Q: %.2f' % (p, d, q))

for t in range(len(test)):
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test, label='Real Value')
pyplot.plot(predictions, label='Predicted Value', color='red')
pyplot.legend()
pyplot.title('Arima - Avg Bitcoin Prices')
pyplot.xlabel('Days')
pyplot.ylabel('USD')
pyplot.show()
