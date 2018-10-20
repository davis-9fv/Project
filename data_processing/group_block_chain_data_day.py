from Util import algorithm
import pandas as pd
import datetime
import numpy as np

write_file = True
path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin.csv'
output_file = 'bitcoin_block_chain_by_day.csv'
df = pd.read_csv(path + input_file, header=0, sep=',')
df_date = df['time']
df_date = df_date.values

columns = ['transaction_count'
    , 'input_count'
    , 'output_count'
    , 'input_total'
    , 'input_total_usd'
    , 'output_total'
    , 'output_total_usd'
    , 'fee_total'
    , 'fee_total_usd'
    , 'generation'
    , 'reward'
    , 'size'
    , 'weight'
    , 'stripped_size']

year_list = list()
df_list = list()
for i in range(len(df_date)):
    date = datetime.datetime.strptime(df_date[i], '%Y-%m-%d %H:%M:%S')
    year_list.append(datetime.date.strftime(date, '%Y-%m-%d'))

year_column = np.asarray(year_list)
df_day = pd.DataFrame({'day': year_column})
df_concat = pd.concat([df, df_day], axis=1)

dfx = pd.DataFrame()
for i in range(len(columns)):
    df_group = df_concat.groupby('day')[columns[i]].sum()
    dfx[columns[i]] = df_group

# df_group2 = df_concat.groupby('day')['input_count'].sum()
# dfx['input_count'] = df_group2


print(dfx)
print(dfx.index.values)

if write_file:
    dfx.to_csv(path + output_file, mode='a', header=True)
