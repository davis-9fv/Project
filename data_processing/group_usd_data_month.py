from Util import algorithm
import pandas as pd
import datetime
import numpy as np

write_file = True
path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin_usd_11_10_2018.csv'
output_file = 'bitcoin_usd_by_month.csv'
df = pd.read_csv(path + input_file, header=0, sep=',')
df_date = df['Date']
df_date = df_date.values

# We drop column volume because it is not complete.
columns = ['Open'
    , 'High'
    , 'Low'
    , 'Close'
    , 'Market Cap'
    , 'Avg']

month_list = list()
df_list = list()
for i in range(len(df_date)):
    date = datetime.datetime.strptime(df_date[i], '%Y-%m-%d')
    month_list.append(datetime.date.strftime(date, '%Y-%m'))

month_column = np.asarray(month_list)
df_month = pd.DataFrame({'month': month_column})
df_concat = pd.concat([df, df_month], axis=1)

dfx = pd.DataFrame()
print(df_concat)
for i in range(len(columns)):
    df_group = df_concat.groupby('month')[columns[i]].mean()
    dfx[columns[i]] = df_group

print(dfx)
print(dfx.index.values)

if write_file:
    dfx.to_csv(path + output_file, mode='a', header=True)
