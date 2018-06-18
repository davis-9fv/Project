from pandas import DataFrame


# Creates a sequence as a supervised learning problem
def timeseries_to_supervised(data, column_name):
    df = DataFrame(data)
    df['output'] = df[column_name]
    df['output'] = df['output'].shift(1)
    df.fillna(0, inplace=True)
    return df


