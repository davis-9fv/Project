import pandas as pd
from pandas import DataFrame
from pandas import concat


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, column_name):
    df = DataFrame(data)
    df['output'] = df[column_name]
    df['output'] = df['output'].shift(1)
    df.fillna(0, inplace=True)
    return df


supervised = False

# load dataset
df = pd.read_csv('../Bitcoin-Historical-Data_28april2013-23april2018.csv', sep='\t')

# print(df.sort_values('DOB'))
df['DateSorted'] = pd.to_datetime(df['Date'])
df = df.sort_values('DateSorted')

# df['Year'] = pd.DatetimeIndex(df['Date']).year
# df['Month'] = pd.DatetimeIndex(df['Date']).month
# df['Day'] = pd.DatetimeIndex(df['Date']).day

df['High'] = df['High'].str.replace(',', '')
df['Low'] = df['Low'].str.replace(',', '')
df[['High', 'Low']] = df[['High', 'Low']].astype('float')
df['Avg'] = (df['High'] + df['Low']) / 2

# df = df.drop(['Date', 'DateSorted', 'High', 'Low', 'Volume', 'Market Cap', 'Open', 'Close'], axis=1)
df = df.drop(['DateSorted', 'High', 'Low', 'Volume', 'Market Cap', 'Open', 'Close'], axis=1)
# df = df.drop(['Volume', 'Market Cap'], axis=1)
# df = df.drop(['Day', 'Year','Month'], axis=1)

if supervised:
    df = timeseries_to_supervised(df, 'Avg')
    # We drop the first row
    df = df.iloc[1:]
    print(df)
    df.to_csv("Bitcoin_historical_data_processed_supervised.csv", sep='\t', encoding='utf-8', index=False)
else:
    print(df)
    df.to_csv("Bitcoin_historical_data_processed_1f.csv", sep='\t', encoding='utf-8', index=False)
