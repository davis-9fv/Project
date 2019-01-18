import pandas as pd
import datetime
import numpy as np

# https://trends.google.com/trends/explore?q=bitcoin

path = 'C:/tmp/bitcoin/'
input_file = 'sources/bitcoin_trend.csv'
output_file = 'bitcoin_trend.csv'
df = pd.read_csv(path + input_file, header=0, sep=',')
write_file = True

start = datetime.datetime.strptime("27-10-2013", "%d-%m-%Y")
end = datetime.datetime.strptime("10-10-2018", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

trend = [0 for x in range(0, len(date_generated))]
for i in range(0, len(date_generated)):
    day = date_generated[i].strftime("%d-%m-%Y")
    print(day)

    for y in range(0, df.shape[0]):
        week_csv = df['Week'].iloc[y]
        week_csv = datetime.datetime.strptime(week_csv, '%Y-%m-%d')
        week_csv = datetime.date.strftime(week_csv, "%d-%m-%Y")
        if week_csv == day:
            trend[i] = df['bitcoin'].iloc[y]
            break
        else:
            trend[i] = 0

print(trend)

print(np.log(4))


def get_value(diff, percentage, start):
    ans = (diff * percentage / 100) + start
    print(ans)
    return ans


data = trend
new_data = data

percentage_1 = 14.29
percentage_2 = 28.58
percentage_3 = 42.87
percentage_4 = 57.16
percentage_5 = 71.45
percentage_6 = 85.74

# sub_data = data[0:8]
# sub_data = data[7:15]
start_block = 0
end_block = 8

while len(data) >= end_block:
    sub_data = data[start_block:end_block]
    print(sub_data)
    start = sub_data[0]
    end = sub_data[len(sub_data) - 1]
    diff = end - start
    value_1 = get_value(diff, percentage_1, start)
    value_2 = get_value(diff, percentage_2, start)
    value_3 = get_value(diff, percentage_3, start)
    value_4 = get_value(diff, percentage_4, start)
    value_5 = get_value(diff, percentage_5, start)
    value_6 = get_value(diff, percentage_6, start)

    new_data[start_block + 1] = ('%.2f') % value_1
    new_data[start_block + 2] = ('%.2f') % value_2
    new_data[start_block + 3] = ('%.2f') % value_3
    new_data[start_block + 4] = ('%.2f') % value_4
    new_data[start_block + 5] = ('%.2f') % value_5
    new_data[start_block + 6] = ('%.2f') % value_6

    start_block = end_block - 1
    end_block = start_block + 8

print(new_data)

dfx = pd.DataFrame()
dfx['Date'] = date_generated
dfx['Trend'] = new_data

print(dfx)

if write_file:
    # The rows from input_block_chain_file are reversed
    dfx = dfx[::-1]
    dfx.to_csv(path + output_file, header=True, index=False)
