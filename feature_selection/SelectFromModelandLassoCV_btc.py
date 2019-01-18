# Author: Manoj Kumar <mks542@nyu.edu>
# License: BSD 3 clause

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from pandas import read_csv
from pandas import concat
from pandas import DataFrame

path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin_usd_bitcoin_block_chain_trend_by_day.csv'

# Load the boston dataset.
series = read_csv(path + input_file, header=0, sep=',', nrows=1438)
series = series.iloc[::-1]


columns = ['Open',
           'High',  'Trend', 'generation','week_of_year_column','Low','reward']
columns = ['Open',
           'High', 'Low', 'Close', 'day_of_week', 'day_of_month', 'day_of_year', 'month_of_year',
           'year', 'week_of_year_column', 'transaction_count', 'input_count', 'output_count',
           'input_total', 'input_total_usd', 'output_total', 'output_total_usd', 'fee_total',
           'fee_total_usd', 'generation', 'reward', 'size', 'weight', 'stripped_size', 'Trend']

dfx = DataFrame()
for column in columns:
    dfx[column] = series[column]


print(dfx.head(1))
X, y = dfx.values, series['Avg'].values

alphas = np.logspace(-10, 1.5, num=100)
# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV(cv=10, max_iter=1800000000,
              alphas=alphas,
              n_jobs=2,
              normalize=True,
            verbose=True)

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.0000001)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
print('Features before:')
print(n_features)

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 10:
    sfm.threshold += 0.0001
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

print(n_features)

# Plot the selected two features from X.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()

print(X_transform[0, :])
