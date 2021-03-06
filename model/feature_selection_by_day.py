# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
# https://machinelearningmastery.com/feature-selection-machine-learning-python/
# Do with PCA, Genetic Algorithm, Decision Tree,Mutual Information
# Variance Threshold
# Wrapper Methods: Forward Search, Recursive Feature Elimination
# https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2
from plistlib import Data

import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pandas import read_csv
from pandas import DataFrame

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_regression
import numpy as np

write_file = True
output_file = 'feature_selection.csv'

path = 'C:/tmp/bitcoin/'
# input_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'
output_file = 'feature_selection.csv'

series = read_csv(path + input_file, header=0, sep=',')

x_series = DataFrame({
    'Open': series['Open'],
    'High': series['High'],
    'Low': series['Low'],
    'Close': series['Close'],
    # 'Volume': series['Volume'],
    'day_of_week': series['day_of_week'],
    'day_of_month': series['day_of_month'],
    'day_of_year': series['day_of_year'],
    'month_of_year': series['month_of_year'],
    'year': series['year'],
    'week_of_year_column': series['week_of_year_column'],

    'transaction_count': series['transaction_count'],
    'input_count': series['input_count'],
    'output_count': series['output_count'],
    'input_total': series['input_total'],
    'input_total_usd': series['input_total_usd'],
    'output_total': series['output_total'],
    'output_total_usd': series['output_total_usd'],
    'fee_total': series['fee_total'],
    'fee_total_usd': series['fee_total_usd'],
    'generation': series['generation'],
    'reward': series['reward'],

    'size': series['size'],
    'weight': series['weight'],
    'stripped_size': series['stripped_size'],
    'Trend': series['Trend']
}
    , dtype='int64')
y_series = DataFrame({'Avg': series['Avg']}, dtype='int')

x, y = x_series.values, y_series.values
# print(x_series.head())
# print(y)
# feature extraction

print(' ')
print('SelectKBest: ')
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(x, y)
# summarize scores
numpy.set_printoptions(precision=2, suppress=True)
print(list(x_series.columns.values))
print(fit.scores_)
features = fit.transform(x)
print(features[0:6, :])

print(' ')
print('ExtraTreesClassifier: ')
model = ExtraTreesClassifier(n_estimators=100)
model.fit(x, y.ravel())
print(list(x_series.columns.values))
print(model.feature_importances_)

print(' ')
print('F_regression: ')
f_test, _ = f_regression(x, y.ravel())
print(list(x_series.columns.values))
print(f_test)

result = np.array([list(x_series.columns.values), fit.scores_, model.feature_importances_, f_test])

if write_file:
    df = DataFrame({'Columns': x_series.columns.values,
                    'SelectKBest': fit.scores_,
                    'ExtraTreesClassifier': model.feature_importances_,
                    'F_regression': f_test})
    df.to_csv(path + output_file, header=True)
