# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
# https://machinelearningmastery.com/feature-selection-machine-learning-python/
# Do with PCA, Genetic Algorithm, Decision Tree,Mutual Information
# Variance Threshold
# Wrapper Methods: Forward Search, Recursive Feature Elimination
# https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2
from pandas import read_csv
from pandas import DataFrame
from sklearn.feature_selection import f_regression
import config

write_file = True
output_file = 'feature_selection.csv'

path = config.selected_path
input_file = config.input_file_ds
output_file = 'feature_selection.csv'

series = read_csv(path + input_file, header=0, sep=',', nrows=1438)



x_series = DataFrame({
    'Open': series['Open'],
    'High': series['High'],
    'Low': series['Low'],
    'Close': series['Close'],
    # The records of the volume column are incomplete
    'Volume': series['Volume'],
    'Market_Cap': series['Market_Cap'],
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
    # 'weight': series['weight'],
    # 'stripped_size': series['stripped_size'],
    'Trend': series['Trend']
}
    , dtype='int64')
y_series = DataFrame({'Avg': series['Avg']}, dtype='int')

x, y = x_series.values, y_series.values

print(' ')
print('F_regression: ')
f_test, _ = f_regression(x, y.ravel())
print(list(x_series.columns.values))
print(f_test)

if write_file:
    df = DataFrame({'Columns': x_series.columns.values,
                    'F_regression': f_test})
    df.to_csv(path + output_file, header=True)
