import pandas as pd
import datetime
import numpy as np

write_file = True
path = 'C:/tmp/bitcoin/'
input_block_chain_file = 'bitcoin_block_chain_by_month.csv'
input_usd_file = 'bitcoin_usd_by_month.csv'
output_file = 'bitcoin_usd_bitcoin_block_chain_by_month.csv'

df_block_chain = pd.read_csv(path + input_block_chain_file, header=0, sep=',')
df_usd = pd.read_csv(path + input_usd_file, header=0, sep=',')

# The rows from input_block_chain_file are reversed
df_block_chain = df_block_chain[::-1]

# print(df_block_chain.head(5))
# print(df_usd.head(5))

print(df_block_chain['month'].max())
print(df_block_chain['month'].min())

print(df_usd['month'].max())
print(df_usd['month'].min())

max_date = df_block_chain['month'].max()
min_date = df_usd['month'].min()


df_result = pd.merge(df_usd, df_block_chain, on='month', how='inner')
print(df_result)

if write_file:
    df_result.to_csv(path + output_file, mode='a', header=True)
