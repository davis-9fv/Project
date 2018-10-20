import pandas as pd
import datetime
import numpy as np

write_file = True
path = 'C:/tmp/bitcoin/'
input_block_chain_file = 'bitcoin_block_chain_by_day.csv'
input_usd_file = 'bitcoin_usd_11_10_2018.csv'
output_file = 'bitcoin_usd_bitcoin_block_chain_by_day.csv'

df_block_chain = pd.read_csv(path + input_block_chain_file, header=0, sep=',')
df_usd = pd.read_csv(path + input_usd_file, header=0, sep=',')

# The rows from input_block_chain_file are reversed
df_block_chain = df_block_chain[::-1]

# print(df_block_chain.head(5))
# print(df_usd.head(5))

print(df_block_chain['day'].max())
print(df_block_chain['day'].min())

print(df_usd['Date'].max())
print(df_usd['Date'].min())

max_date = df_block_chain['day'].max()
min_date = df_usd['Date'].min()

df_block_chain['Date'] = df_block_chain['day']

df_result = pd.merge(df_usd, df_block_chain, on='Date', how='inner')
print(df_result)

if write_file:
    df_result.to_csv(path + output_file, mode='a', header=True)
