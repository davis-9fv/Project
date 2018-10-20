from Util import algorithm
import pandas as pd
import datetime
import numpy as np

df = pd.read_csv('../data/airline-passengers.csv', header=0, sep='\t')
df_date = df['Date']
df_date = df_date.values

year_list = list()
for i in range(len(df_date)):
    date = datetime.datetime.strptime(df_date[i], '%Y-%m')
    year_list.append(datetime.date.strftime(date, "%Y"))

year_column = np.asarray(year_list)
df_year = pd.DataFrame({'year': year_column})
df_result = pd.concat([df, df_year], axis=1)

df_group = df_result.groupby('year')['Passangers'].sum()


print(df_group)
