import pandas as pd
#pip install geopy
from geopy.geocoders import Nominatim 
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

def encode_tags(df_original, min_to_drop = 0):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        df_original (DataFrame): the dataframe that we are encoding
        Integer - minimum number of times a tag must appear to be kept

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    df = df_original[['property_id', 'tags']].copy()
    df_original = df_original.drop('tags', axis=1)
    df_original = df_original.drop_duplicates()
    df = df.explode('tags')
    df = df.drop_duplicates().reset_index(drop=True)
    df = pd.get_dummies(df, columns=['tags'], dummy_na=True)
    df = df.groupby('property_id').sum().reset_index()
    column_sums = df.drop('property_id', axis=1).sum()
    small_columns = column_sums[column_sums <= min_to_drop]
    small_columns_list = small_columns.index.tolist()
    df = df.drop(small_columns_list, axis=1)
    df = pd.merge(df_original, df, how='left', on='property_id')

    return df

def break_it_down(column):
    """Use this to break down a column that has multiple columns inside it
    
    Args:
        column (numpy array): the column to be broken down
    
    Returns:
        
    """
    col_dict = column.to_dict()
    col_df = pd.DataFrame(col_dict).transpose()
    return col_df


def item_replacement(dict, df, col_main, col_1_to_replace, col_2_to_replace):
    """Use this function to iterate through a dictionary replacing
      all the nulls in 2 seperate columns based off of the contents
        of a the main column
      
      Args:
            dict: A Dictionary with 1 column that will be used to replace 2 others
            df (DataFrame): the input dataframe
            col_main (string): The main column, no data changed in here but based on this column other columns will be replaced
            col_1_to_replace (string): the first column of data we are filling nulls in
            col_2_to_replace (string): the second column of data we are filling nulls in
      
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
            dict: The dictionary of citys, types and the sqft and lot_sqft
            pandas.Dataframe: the orignal dataframe to adjust
      Returns:
            pandas.Dataframe: modified with nulls added
      """
    for (type, city), numbers in dict.items():
        df.loc[(df['type'] == type) & (df['city'] == city) & (df['sqft'].isna()), 'sqft'] = numbers['sqft']
        df.loc[(df['type'] == type) & (df['city'] == city) & (df['lot_sqft'].isna()), 'lot_sqft'] = numbers['lot_sqft']
    return df



def get_downtown_coordinates(df, city_col, state_col, batch_size=10, max_retries=3): 
    """Gets latitude and longitude for downtown regions of city/state combinations from DataFrame columns,
       avoiding duplicate lookups.

    Args:
        df (pd.DataFrame): DataFrame containing city and state columns.
        city_col (str): Name of the column containing city names.
        state_col (str): Name of the column containing state/region names.
        batch_size (int): The number of unique city/state combinations to process in each batch.

    Returns:
        pd.DataFrame: The original DataFrame with added 'latitude' and 'longitude' columns.
    """

    unique_city_states = set(df[city_col] + ", " + df[state_col])
    geolocator = Nominatim(user_agent="downtown_locator")
    coordinates = {}
    df['city_lat'] = None
    df['city_lon'] = None

    unique_city_states_list = list(unique_city_states)
    remaining_attempts = unique_city_states_list[:]  # Start with all cities

    for retry_num in range(max_retries + 1):  # Loop for retries (including the initial attempt)
        batch_processing = remaining_attempts[:] # Copy remaining attempts for batch processing
        remaining_attempts = [] # reset the remaining attempts
        print(f"Geocoding attempt {retry_num + 1} of {max_retries + 1}...")

        for i in range(0, len(batch_processing), batch_size):
            batch = batch_processing[i:i + batch_size]
            for city_state in batch:
                if city_state not in coordinates:
                    try:
                        location = geolocator.geocode(city_state + ", city centre")
                        if not location:
                            location = geolocator.geocode(city_state + ", downtown")
                        if not location:
                            location = geolocator.geocode(city_state)

                        if location:
                            coordinates[city_state] = (location.latitude, location.longitude)
                            #print(f"Found coordinates for {city_state}: {location.latitude}, {location.longitude}")
                        else:
                            print(f"Could not find coordinates for {city_state}")
                            remaining_attempts.append(city_state)  # Add to retry list

                    except (GeocoderTimedOut, GeocoderServiceError) as e:
                        print(f"Error geocoding {city_state}: {e}")
                        remaining_attempts.append(city_state)  # Add to retry list
                        time.sleep(1) # Respect rate limits

                    except Exception as e:
                        print(f"An unexpected error occurred for {city_state}: {e}")
                        remaining_attempts.append(city_state)  # Add to retry list

                time.sleep(1)  # Pause between batches

        if not remaining_attempts: # if remaining_attempts is empty then break out of the loop
            break

    # Merge coordinates back into the DataFrame (same as before)
    for index, row in df.iterrows():
        city_state = row[city_col] + ", " + row[state_col]
        if city_state in coordinates:
            df.loc[index, 'city_lat'] = coordinates[city_state][0]
            df.loc[index, 'city_lon'] = coordinates[city_state][1]

    return df