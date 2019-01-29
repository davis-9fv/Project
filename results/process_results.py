import datetime
from pandas import DataFrame
from pandas import read_csv
import numpy as np

time_start = datetime.datetime.now()
print('Start time: %s' % str(time_start.strftime('%Y-%m-%d %H:%M:%S')))

path = 'C:/tmp/bitcoin/'
# path = '/home/fran_vinueza/'
input_file = 'main_window_x_size_results2.csv'

# To pair with the other models, this model gets 1438 first rows.
data = read_csv(path + input_file, header=0, sep=',')

# columns = ['Avg Elastic', 'Avg Lasso', 'Avg KNN5', 'Avg KNN10', 'Avg SGD', 'Avg LSTM']
columns = ['Sum Elastic', 'Sum Lasso', 'Sum KNN5', 'Sum KNN10', 'Sum SGD' ]

results = DataFrame(index=data.index.values, columns=columns)
results = results.fillna(0)

results['Sum Elastic'] = data['(Tr) ElasticNet'] + data['(Te) ElasticNet']
results['Sum Lasso'] = data['(Tr) Lasso'] + data['(Te) Lasso']
results['Sum KNN5'] = data['(Tr) KNN5'] + data['(Te) KNN5']
results['Sum KNN10'] = data['(Tr) KNN10'] + data['(Te) KNN10']
results['Sum SGD'] = data['(Tr) SGD'] + data['(Te) SGD']
results['Min Value'] = results.min(axis=1)

elastic_min = results['Sum Elastic'].min()
index_elastic_min = results['Sum Elastic'].idxmin()
lasso_min = results['Sum Lasso'].min()
index_lasso_min = results['Sum Lasso'].idxmin()
knn5_min = results['Sum KNN5'].min()
index_knn5_min = results['Sum KNN5'].idxmin()
knn10_min = results['Sum KNN10'].min()
index_knn10_min = results['Sum KNN10'].idxmin()
sgd_min = results['Sum SGD'].min()
index_sgd_min = results['Sum SGD'].idxmin()

best_row = results['Min Value'].idxmin()

print('Lowest Sum Elastic  %.3f' % (elastic_min))
print("Lowest Sum Lasso    %.3f" % (lasso_min))
print("Lowest Sum KNN5     %.3f" % (knn5_min))
print("Lowest Sum KNN10    %.3f" % (knn10_min))
print("Lowest Sum SGD      %.3f" % (sgd_min))

print('From Elastic, Row:  %i' % (index_elastic_min))
print("From Lasso, Row:    %i" % (index_lasso_min))
print("From KNN5, Row:     %i" % (index_knn5_min))
print("From KNN10, Row:    %i" % (index_knn10_min))
print("From SGD, Row:      %i" % (index_sgd_min))

min_results_col = [elastic_min, lasso_min, knn5_min, knn10_min, sgd_min]
index_col_min = np.asarray(min_results_col).argmin()
print("Best column:  %s" % (columns[index_col_min]))

print(results)
print("Best row: %i" % (best_row))
print("Best combination:")
print(data.iloc[best_row])