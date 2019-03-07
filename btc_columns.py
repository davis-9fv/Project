bitcoin_columns_opt_test = [
    ['year', 'generation', 'input_count', 'size', 'fee_total', 'reward', 'output_count', 'weight'],
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
     'fee_total', 'reward', 'output_count', 'weight', 'transaction_count',
     'stripped_size', 'output_total', 'input_total', 'week_of_year_column',
     'day_of_year', 'month_of_year', 'day_of_month', 'day_of_week']
]

bitcoin_columns_opt_pro = [
    # All features
    ['High', 'Low', 'Close', 'Open', 'Market_Cap', 'output_total_usd', 'input_total_usd',
     'Trend', 'year', 'fee_total_usd', 'generation', 'input_count',
     'fee_total', 'reward', 'output_count', 'weight', 'transaction_count',
     'stripped_size', 'output_total', 'input_total', 'week_of_year_column',
     'day_of_year', 'month_of_year', 'day_of_month', 'day_of_week'],

    # 12 best f_regression
    ['High', 'Low', 'Close', 'Open', 'Market_Cap', 'output_total_usd', 'input_total_usd',
     'Trend', 'year', 'fee_total_usd', 'generation', 'input_count'],

    # 8 best f_regression
    ['High', 'Low', 'Close', 'Open', 'Market_Cap', 'output_total_usd', 'input_total_usd',
     'Trend'],

    # 4 best f_regression
    ['High', 'Low', 'Close', 'Open'],

    # 8 best f_regression no usd
    ['Trend', 'year', 'generation', 'input_count', 'size', 'fee_total',
     'reward', 'output_count', 'weight'],

    # 4 best f_regression no usd
    ['Trend', 'year', 'generation', 'input_count']

]
