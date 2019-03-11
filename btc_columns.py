bitcoin_columns_opt_test = [
    ['year', 'generation', 'input_count', 'size', 'fee_total', 'reward', 'output_count'],
    ['year', 'generation']
]

bitcoin_columns_opt_1_col = [
    ['year']
]

bitcoin_columns_opt_2_col = [
    ['input_count', 'size']
]

bitcoin_columns_opt_all = [
    # All features
    ['High', 'Low', 'Close', 'Open', 'Market_Cap', 'output_total_usd', 'input_total_usd',
     'Trend', 'year', 'fee_total_usd', 'generation', 'input_count',
     'fee_total', 'reward', 'output_count', 'transaction_count',
     'output_total', 'input_total', 'week_of_year_column',
     'day_of_year', 'month_of_year', 'day_of_month', 'day_of_week']
]

bitcoin_columns_opt_pro = [

    # 12 best f_regression
    ['High', 'Low', 'Close', 'Open', 'Market_Cap', 'Volume', 'output_total_usd', 'input_total_usd',
     'Trend', 'year', 'fee_total_usd', 'generation'],

    # 8 best f_regression
    ['High', 'Low', 'Close', 'Open', 'Market_Cap', 'Volume' ,'output_total_usd', 'input_total_usd'],

    # 4 best f_regression
    ['High', 'Low', 'Close', 'Open'],

    # 8 best f_regression no usd
    ['Trend', 'year', 'input_count', 'size', 'generation', 'fee_total',
     'reward', 'output_count'],

    # 4 best f_regression no usd
    ['Trend', 'year', 'input_count', 'generation']

]
