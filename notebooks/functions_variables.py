import pandas as pd

def encode_tags(df, min_to_drop = 0):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame
        Integer - minimum number of times a tag must appear to be kept

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    df = df.explode('tags')
    df = df.drop_duplicates().reset_index(drop=True)
    df = pd.get_dummies(df, columns=['tags'], dummy_na=True)
    df = df.groupby('property_id').sum().reset_index()
    column_sums = df.drop('property_id', axis=1).sum()
    small_columns = column_sums[column_sums < 20]
    small_columns_list = small_columns.index.tolist()
    df = df.drop(small_columns_list, axis=1)
    return df


#run through the dictionary of cities and the average longitutde/latitude for them and return 
def item_replacement(dict, df, col_main, col_1_to_replace, col_2_to_replace):
    """Use this function to iterate through a dictionary replacing
      all the nulls in 2 seperate columns based off of the contents
        of a the main column
      
      Args:
            dictionary
            pandas.Dataframe
            col_main = string
            col_1_to_replace = string
            col_2_to_replace = string
      
      Returns:
            pandas.Dataframe: modified with nulls added
      """
    for main_column, replacement_columns in dict.items():
        df.loc[(df[col_main] == main_column) & (df[col_1_to_replace].isna()), col_1_to_replace] = replacement_columns[col_1_to_replace]
        df.loc[(df[col_main] == main_column) & (df[col_2_to_replace].isna()), col_2_to_replace] = replacement_columns[col_2_to_replace]
    return df



def sqfts_replacement(dict, df):
    """Use this function to iterate through a dictionary replacing
      all the nulls in 2 seperate columns based off of the contents
        of 2 main columns
      
      Args:
            dictionary
            pandas.Dataframe
      Returns:
            pandas.Dataframe: modified with nulls added
      """
    for (type, city), numbers in dict.items():
        df.loc[(df['type'] == type) & (df['city'] == city) & (df['sqft'].isna()), 'sqft'] = numbers['sqft']
        df.loc[(df['type'] == type) & (df['city'] == city) & (df['lot_sqft'].isna()), 'lot_sqft'] = numbers['lot_sqft']
    return df

