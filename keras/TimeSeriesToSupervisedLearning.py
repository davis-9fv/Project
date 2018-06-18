from bokeh.sphinxext.collapsible_code_block import collapsible_code_block
from pandas import DataFrame
from pandas import Series
from pandas import concat

# frame a sequence as a supervised learning problem
from scipy.special._ufuncs import shichi


def timeseries_to_supervised(data, column_name):
    df = DataFrame(data)
    df['output'] = df[column_name]
    df['output'] = df['output'].shift(1)
    df.fillna(0, inplace=True)
    return df


df = DataFrame()
df['A'] = [10, 20, 30, 40, 50]
df['B'] = ["ES", "EC", "US", "UK", "CO"]
# print(df)

column_names = ['AA', 'BB']
# df = df.rename(index=str, columns={'A': 'a', 'B': 'b'})

df = timeseries_to_supervised(df, 'A')
print(df)
