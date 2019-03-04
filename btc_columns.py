bitcoin_columns_opt_test = [
    ['year', 'generation', 'input_count', 'size', 'fee_total', 'reward', 'output_count', 'weight'],
    ['year', 'generation']
]

bitcoin_columns_opt_1_col = [
    ['year']
]

bitcoin_columns_opt_2_col = [
    ['input_count','size']
]

bitcoin_columns_opt_all = [
    # All features
    ['Open', 'High', 'Low', 'Close', 'day_of_week', 'day_of_month', 'day_of_year', 'month_of_year', 'year',
     'week_of_year_column', 'transaction_count', 'input_count', 'output_count', 'input_total', 'input_total_usd',
     'output_total', 'output_total_usd', 'fee_total', 'fee_total_usd', 'generation', 'reward', 'size', 'weight',
     'stripped_size']
]

bitcoin_columns_opt_pro = [
    # All features
    ['Open', 'High', 'Low', 'Close', 'day_of_week', 'day_of_month', 'day_of_year', 'month_of_year', 'year',
     'week_of_year_column', 'transaction_count', 'input_count', 'output_count', 'input_total', 'input_total_usd',
     'output_total', 'output_total_usd', 'fee_total', 'fee_total_usd', 'generation', 'reward', 'size', 'weight',
     'stripped_size'],

    # 9 best f_regression No USD, Low, High, Open High
    ['year', 'generation', 'input_count', 'size', 'fee_total', 'reward', 'output_count', 'weight', 'transaction_count'],

    # 12 best f_regression
    ['Open', 'High', 'Low', 'Close', 'output_total_usd', 'input_total_usd', 'year', 'fee_total_usd', 'generation',
     'input_count', 'size', 'fee_total'],

    # 12 best f_regression, No Low, High, Open High
    ['output_total_usd', 'input_total_usd', 'year', 'fee_total_usd', 'generation', 'input_count', 'size',
     'fee_total', 'reward', 'output_count', 'weight', 'transaction_count'],

    # 8 best f_regression
    ['Open', 'High', 'Low', 'Close', 'output_total_usd', 'input_total_usd', 'year', 'fee_total_usd'],

    # 8 best f_regression, No Low, High, Open High
    ['output_total_usd', 'input_total_usd', 'year', 'fee_total_usd', 'generation', 'input_count', 'size'],

    # 12 best ExtraTreesClassifier
    ['Open', 'High', 'Low', 'Close', 'day_of_year', 'day_of_month', 'reward', 'generation', 'fee_total_usd',
     'output_count', 'output_total_usd', 'size'],

    # 12 best ExtraTreesClassifier, No Low, High, Open High
    ['day_of_year', 'day_of_month', 'reward', 'generation', 'fee_total_usd', 'output_count', 'output_total_usd', 'size',
     'input_count', 'input_total_usd', 'output_total', 'transaction_count'],

    # 8 best ExtraTreesClassifier
    ['Open', 'High', 'Low', 'Close', 'day_of_year', 'day_of_month', 'reward', 'generation', 'fee_total_usd'],

    # 8 best ExtraTreesClassifier, No Low, High, Open High
    ['day_of_year', 'day_of_month', 'reward', 'generation', 'fee_total_usd', 'output_count', 'output_total_usd', 'size',
     'input_count'],

    # Intersection from 12 best
    ['Low', 'High', 'Close', 'Open', 'generation', 'fee_total_usd', 'output_total_usd', 'size'],

    # Intersection from 12 best, No Low, High, Open High
    ['reward', 'generation', 'fee_total_usd', 'output_count', 'output_total_usd', 'size', 'input_count',
     'input_total_usd', 'transaction_count'],

    # Union from 12 best
    ['Low', 'High', 'Close', 'Open', 'generation', 'fee_total_usd', 'output_total_usd', 'size', 'input_total_usd',
     'year', 'input_count', 'day_of_year', 'day_of_month', 'reward', 'output_count', 'fee_total'],

    # Union from 12 best, No Low, High, Open High
    ['reward', 'generation', 'fee_total_usd', 'output_count', 'output_total_usd', 'size', 'input_count',
     'input_total_usd', 'transaction_count', 'output_total_usd', 'year', 'fee_total', 'weight', 'day_of_year',
     'day_of_month', 'output_total_usd', 'output_total'],
]
