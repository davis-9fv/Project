import pandas as pd
import datetime
import numpy as np
import re

path = 'C:/tmp/bitcoin/'
input_file = 'sources/bitcoin_usd_11_10_2018.csv'
output_file = 'bitcoin_usd_11_10_2018.csv'
df = pd.read_csv(path + input_file, header=0, sep=',')
write_file = True

print(df.head(10))
df_date = df['Date']
df_date = df_date.values

# The date column is modified with a new format
new_format_date = list()
for i in range(len(df)):
    date = datetime.datetime.strptime(df_date[i], '%b %d, %Y')
    new_format_date.append(datetime.date.strftime(date, '%Y-%m-%d'))
new_date_column = np.asarray(new_format_date)
df['Date'] = pd.DataFrame({'month': new_date_column})

# Commas are drop from the strings
for i in range(len(df)):
    value = df['Open'].values[i]
    df['Open'].values[i] = re.sub(',', '', value)
    value = df['High'].values[i]
    df['High'].values[i] = re.sub(',', '', value)
    value = df['Low'].values[i]
    df['Low'].values[i] = re.sub(',', '', value)
    value = df['Close'].values[i]
    df['Close'].values[i] = re.sub(',', '', value)
    value = df['Volume'].values[i]
    df['Volume'].values[i] = re.sub(',', '', value)
    value = df['Market Cap'].values[i]
    df['Market Cap'].values[i] = re.sub(',', '', value)

# Column Avg is added
df['Avg'] = (df['High'].apply(lambda x: float(x)) + df['Low'].apply(lambda x: float(x))) / 2
df['Avg'] = df['Avg'].round(2)
print(df.head(10))

if write_file:
    df.to_csv(path + output_file, mode='a', header=True)
