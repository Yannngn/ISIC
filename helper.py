import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def df_groupby_count(df, column):
    return df.groupby(column).count()

def df_reduce_num_classes(df, old, new, min_value=250):
    for group_name, group in df.groupby(old):
        if len(group) < min_value:
            df.loc[df[old] == group_name, old] = group[new].values
        
    return df

def df_group_drop_min_count(df, column, min_value=250):
    for group_name, group in df.groupby(column):
        if len(group) < min_value:
            df = df.loc[df[column] != group_name]
        
    return df

def df_sample_by_column_value(df, column, min_value=250):
    grouped = df.groupby(column)

    # Filter out groups that have less than n_samples
    grouped_filtered = grouped.filter(lambda x: len(x) >= min_value)

    # Sample the same number of items from each group
    subset = grouped_filtered.groupby(column).apply(lambda x: x.sample(n=min_value))

    # Reset the index of the resulting subset
    return subset.reset_index(drop=True)