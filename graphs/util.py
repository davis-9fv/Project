def get_first_row(data, win_size, column):
    data = data.loc[data['Window Size'] == win_size]
    data = data.sort_values(by=[column])
    first_row = data.iloc[0]
    return first_row.values


def get_rows(data, win_size, column):
    data = data.loc[data['Window Size'] == win_size]
    data = data.sort_values(by=[column])
    return data


def get_first_row_alpha(data, win_size, column, alpha):
    data = data.loc[(data['Window Size'] == win_size) & (data['Alpha'] == alpha)]
    data = data.sort_values(by=[column])
    first_row = data.iloc[0]
    return first_row.values


def filter_rmse(data, column):
    # columns = data.columns.values
    rows = []
    for win_size in range(3, 13):
        row = get_first_row(data, win_size, column)
        rows.append(row)

    # filtered_df = pd.DataFrame(rows, columns=columns)
    return rows
