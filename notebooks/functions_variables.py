import pandas as pd
#pip install geopy
from geopy.geocoders import Nominatim 
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import time
import numpy as np
import os
import json
import itertools

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
    boolean_cols = df.filter(regex='^tag').columns
    df[boolean_cols] = df[boolean_cols].astype('bool')

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

    # Merge coordinates back into the DataFrame
    for index, row in df.iterrows():
        city_state = row[city_col] + ", " + row[state_col]
        if city_state in coordinates:
            df.loc[index, 'city_lat'] = coordinates[city_state][0]
            df.loc[index, 'city_lon'] = coordinates[city_state][1]

    return df

def clean_data(df, type_to_drop = 20):
    """Use this to clean all the data from raw data to the final product
     - This is the EDA cleaning section all combined down into one function, for fully commented and broken down read through EDA sheet

    Args:
    df (pandas.DataFrame)

    Returns:
    pd.DataFrame cleaned up and ready for machine learning
    """
    columns_to_drop = ['primary_photo',
                   'last_update_date',
                     'source', 
                      'permalink',
                        'status',
                          'list_date',
                           'open_houses',
                            'branding',
                             'list_price',
                              'lead_attributes',
                                'photos',
                                'virtual_tours',
                                'other_listings',
                                 'listing_id',
                                  'price_reduced_amount',
                                   'matterport',
                                    'sold_date',
                                     'products',
                                      'street_view_url',
                                       'community',
                                        'county',
                                         'line',
                                           'flags',
                                            'name',
                                             'baths_1qtr',
                                              'sub_type',
                                               'baths_full',
                                                'baths_half',
                                                 'baths_3qtr',
                                                  'state_code',
                                                   'description',
                                                    'location',
                                                     'address',
                                                      'coordinate',
                                                      ]
    #break down the columns that have lists inside them
    desc_df = break_it_down(df['description'])
    df[desc_df.columns] = desc_df
    loc_df = break_it_down(df['location'])
    df[loc_df.columns] = loc_df
    address_df = break_it_down(df['address'])
    df[address_df.columns] = address_df
    coordinate_df = break_it_down(df['coordinate'])
    df[coordinate_df.columns] = coordinate_df
    #drop rows that are missing the "sold_price", we can't learn from them, drop unneccesary columns
    df = df.dropna(subset=['sold_price'])
    df = df.drop(columns=columns_to_drop, axis=1)
    #various techniques to fill null values
    df['garage'] = df['garage'].fillna(0)
    lon_lat_dict = df[['city', 'lon', 'lat']].groupby('city').mean().transpose().to_dict()
    df = item_replacement(lon_lat_dict, df, 'city', 'lon', 'lat')
    missing_property_dict = {'Boone': {'lon': -93.885490, 'lat': 42.060650},
                'Garnett': {'lon': 81.2454, 'lat': 32.6063},
                'Charlton Heights': {'lon': -81.24385, 'lat': 38.13673}}
    df = item_replacement(missing_property_dict, df, 'city', 'lon', 'lat')
    df = df.rename(columns={'lat':'property_lat', 'lon': 'property_lon'})
    df['city'] = df['city'].fillna('Columbus')
    type_mapping = {'other': 'land',
                'condos': 'condo',
                np.nan : 'land',
                }
    df['type'] = df['type'].replace(type_mapping)
    df = df[(df['type'] != 'condo_townhome_rowhome_coop') & (df['type'] != 'duplex_triplex')]
    to_change_list = ['year_built', 'sqft', 'baths', 'stories', 'beds']
    for col in to_change_list:
        df.loc[(df['type'] == 'land') & (df[col].isna()), col] = 0 
    mean_year = df['year_built'].mean().astype(int)
    df['year_built'] = df['year_built'].fillna(mean_year)
    bed_bath_dict = df[['type', 'beds', 'baths']].groupby('type').mean().astype(int).transpose().to_dict()
    df = item_replacement(bed_bath_dict, df, 'type', 'beds','baths')
    df['stories'] = df['stories'].fillna(1)
    sqfts_dict = df[['city', 'type', 'sqft', 'lot_sqft']].groupby(['type','city']).mean().transpose().to_dict()
    df = sqfts_replacement(sqfts_dict, df)
    sqfts_dict_2 = df[['type', 'sqft', 'lot_sqft']].groupby('type').mean().transpose().to_dict()
    df = item_replacement(sqfts_dict_2, df, 'type', 'sqft', 'lot_sqft')
    #split out and hot hit encode the tags
    df = encode_tags(df, type_to_drop)
    #replace cities with lon/lat
    df = get_downtown_coordinates(df=df,city_col='city',state_col='state')
    #correc the datatypes
    int_columns = ['year_built', 'sqft', 'lot_sqft', 'baths','garage','stories','beds', 'postal_code', 'sold_price']
    df[int_columns] = df[int_columns].astype(int)
    category_columns = ['type']
    df[category_columns] = df[category_columns].astype('category')
    float_columns = ['city_lat', 'city_lon', 'property_lon', 'property_lat']
    df[float_columns] = df[float_columns].astype('float')
    #drop more unneccesary columns
    df = df.drop(['city', 'state','property_id'], axis=1)
    #encode the type column
    df = pd.get_dummies(df, columns=['type'], dummy_na=True)
    return df

def load_data(folder_name):
    """Use this to load the data from multiple json files into one dataframe

    Args:
    folder_name(string): the name of the folder containing all the json files

    Returns
    pd.DataFrame: Dataframe containing all the information (unprocessed)
    
    """
    filenames = os.listdir(folder_name)
    df = pd.DataFrame()
    empty_files = []
    #Iterate through every data file we have
    for file in filenames:
        #ensure files are "json" files
        if file.endswith(".json"):
            file_path = os.path.join(folder_name, file)

            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    #create a small dataframe which we will add onto the large one
                    small_df = pd.DataFrame(data['data']['results'])
                    # print(file, "Loaded Sucessfully") - for testing purposes
                    #add the new data to the bottom of our dataframe
                    if small_df.empty:
                        # print("file is empty:", file)
                        empty_files.append(file)
                    else:
                        df = pd.concat([df, small_df], ignore_index = True)
                except json.JSONDecodeError as e:
                    #print if there was an error
                    print("Error Decoding file:", e, file)
        else:
            #print out any files that are not part of it
            print("Not a Json:", file)
    return df

class DataLoader(TransformerMixin):
    """
    Quick class to hold the load data function
    """
    def __init__(self, source_folder_name):
        self.folder_name = source_folder_name

    def fit(self, X=None, y=None):  
        return self  

    def transform(self, X=None): 
        data = load_data(self.folder_name)  
        return data

class DataCleaner(TransformerMixin):
    """
    Quick class to hold the clean_data function
    """
    def __init__(self, num_type_to_drop = 20):
        self.to_drop = num_type_to_drop

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):  
        cleaned_data = clean_data(X, self.to_drop)  
        return cleaned_data
    
class Scalar_numeric(TransformerMixin):
    """
    Takes only the numeric columns and scales them, returns only numeric columns
    """

    def __init__(self):
        self.numeric_columns = None
        self.scalar = StandardScaler()
        
    def fit(self, X, y=None):
        self.numeric_columns = list(X.select_dtypes(include=['int', 'float']).columns)
        self.scalar.fit(X[self.numeric_columns])  
        return self
    
    def transform(self, X, y=None):
        X = X[self.numeric_columns].copy()
        X = self.scalar.transform(X)
        return X
    
class PCA_bool(TransformerMixin):
    """
    Takes only the boolean columns and performs PCA filtering on them down to 10 columns
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.bool_columns = None

    def fit(self, X, y=None):
        self.bool_columns = X.select_dtypes(include=['bool']).columns
        self.pca.fit(X[self.bool_columns])
        return self
    
    def transform(self, X,y=None):
        X = X[self.bool_columns].copy()
        X = self.pca.transform(X)
        return X    

def custom_cross_validation(X_train, y_train, n_splits=5):
    """
    Creates n_splits sets of training and validation folds
    
    Args:         
        X_train(DataFrame): data to be splt and validated
        y_train(series): target series
        n_splits (int): number of times to perform k folds
    """
    training_data = X_train.copy()
    training_data["target"] = y_train  

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  
    training_folds, validation_folds = [], []

    for train_idx, val_idx in kf.split(training_data):
        train_fold = training_data.iloc[train_idx]
        val_fold = training_data.iloc[val_idx]
        
        training_folds.append(train_fold)
        validation_folds.append(val_fold)

    return training_folds, validation_folds

def hyperparameter_search(training_folds, validation_folds, param_grid):
    """
    Searches through the paramaters and finds the best combination

    Args:
        training_folds(dataframe): folds to train model with
        validation_folds(dataframe): folds to test model with
        param_grid(dictionary): set of conditions to test against
    """

    best_score = float('inf')  
    best_params = None  

    # create all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in param_combinations:
        scores = []

        for train_fold, val_fold in zip(training_folds, validation_folds):
            X_train, y_train = train_fold.drop(columns=["target"]), train_fold["target"]
            X_val, y_val = val_fold.drop(columns=["target"]), val_fold["target"]

            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = mean_squared_error(y_val, predictions)
            scores.append(score)

        avg_score = np.mean(scores)

        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    return best_params

def CalculateScores(y_pred, y_test):
    """
    Calculate and return the rmse, mae and r2 for the supplied data

    Args:
        y_pred(series): result of the model prediction
        y_test(series): validation set saved from initial
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.4f}\n")
    return rmse, mae, r2
