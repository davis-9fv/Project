import pandas as pd

from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform

a = [1, 2, 3, 4, 5]
b = [12, 24, 36, 48, 60]
c = [100, 10, 100, 100, 10]

# df = pd.DataFrame({'a': a, 'b': b, 'c': c})


path = 'C:/tmp/bitcoin/'
input_file = 'bitcoin_usd_bitcoin_block_chain_by_month.csv'
dfx = pd.read_csv(path + input_file, header=0, sep=',')

df = pd.DataFrame()
df['transaction_count'] = dfx['transaction_count']
df['input'] = dfx['input_count']
df['output'] = dfx['output_count']
df['inputTotal'] = dfx['input_total']
df['inputTotalUsd'] = dfx['input_total_usd']
df['outputTotal'] = dfx['output_total']
df['outputTotalUsd'] = dfx['output_total_usd']
df['feeTotal'] = dfx['fee_total']
df['feeTotalUsd'] = dfx['fee_total_usd']
df['generation'] = dfx['generation']
df['reward'] = dfx['reward']
df['size'] = dfx['size']
df['weight'] = dfx['weight']
df['stripped_size'] = dfx['stripped_size']

df['Open'] = dfx['Open']
df['High'] = dfx['High']
df['Low'] = dfx['Low']
df['Close'] = dfx['Close']

df['Avg'] = dfx['Avg']
df['Market Cap'] = dfx['Market Cap']

cor_matrix = df.corr(method='pearson', min_periods=1)

print(cor_matrix)
cor_matrix *= 100

output_file('correlation_data_by_month.html')
# print(data.head)
# data.Year = data.Year.astype(str)
# data = data.set_index('Year')
# data.drop('Annual', axis=1, inplace=True)
# data.columns.name = 'Month'

# print(data.head)

# reshape to 1D array or rates with a month and year for each row.
# df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()
# print(df.head)

df = pd.DataFrame(cor_matrix.stack(), columns=['rate']).reset_index()
df.columns = ['items_x', 'items_y', 'value']
print(df.head)

print(cor_matrix.columns)
print(cor_matrix.index)

source = ColumnDataSource(df)

# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=df.value.min(), high=df.value.max())

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

p = figure(plot_width=650, plot_height=450, title="Bitcoin Information by Month",
           x_range=list(cor_matrix.index), y_range=list(reversed(cor_matrix.columns)),
           toolbar_location='below', tools=TOOLS, x_axis_location="above",
           tooltips=[('Item', '@items_x vs @items_y'), ('Rate', '@value%')])

p.rect(x="items_x", y="items_y", width=1, height=1, source=source,
       line_color=None, fill_color=transform('value', mapper))

color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%.2f%%"))

p.add_layout(color_bar, 'right')

p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "8.5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = 1.0

show(p)
