import pandas as pd
import numpy as np
from StringFunctions import replace


###################################################################
# row_selection/column_selection by index
#  Single values: 1
#  Integer list of rows/columns: [0,1,2]
#  Slice of rows/columns: [4:7]
###################################################################
def select_indexed(row_selection, column_selection, df):
    return df.iloc[row_selection, column_selection]


###################################################################
# row_selection/column_selection by name
#  Single values: "first_name"
#  List of rows/columns names: ["first_name", "age"]
#  Slice of rows/columns: ["first_name":"address]
###################################################################
def select_named(row_selection, column_selection, df):
    return df.loc[row_selection, column_selection]


###################################################################
# Build a data frame out of columns
###################################################################
def create_df_by_columns(list_columns, columns_name):
    return pd.concat([pd.Series(x) for x in list_columns], axis=1, keys=columns_name)


###################################################################
# Sort the df with key the lambda function applied to column with index column_idx
###################################################################
def sort_df(df, column_idx, key):
    '''Takes dataframe, column index and custom function for sorting,
    returns dataframe sorted by this column using this function'''

    col = df.ix[:, column_idx]
    temp = np.array(col.values.tolist())
    order = sorted(range(len(temp)), key=lambda j: key(temp[j]))
    return df.ix[order]


###################################################################
# Print the pandas df in a pretty format
###################################################################
def visualize_pandas_df(df, slack=6):
    def create_empty_table(length_per_column, num_rows):
        top = "_"*(sum(length_per_column)+len(length_per_column))
        spaces_per_feature = tuple(map(lambda length: length*" ", length_per_column))
        row = ("|%s"*len(length_per_column) + "|") % spaces_per_feature
        return [top] + [row] + [top] + [row]*(num_rows-1) + [top]

    string_df = df.applymap(str)
    headers = string_df.columns
    features_max_length = map(lambda i: max(len(headers[i]), len(max(string_df.iloc[:,i], key=len)))+slack, range(len(headers)))

    start_index_per_feature = map(lambda i: sum(features_max_length[:i]) + slack/2 + i, range(len(headers)))

    def fill_row(empty_row, list_row):
        for i in range(len(list_row)):
            empty_row = replace(start_index_per_feature[i], list_row[i], empty_row)
        return empty_row

    table = create_empty_table(features_max_length, len(string_df)+1)

    table[1] = fill_row(table[1], headers)
    for i in range(0, len(string_df)):
        table[i+3] = fill_row(table[i+3], string_df.iloc[i,:])

    return reduce(lambda x, y: "%s\n%s" % (x,y), table) + "\n"
