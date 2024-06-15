import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

columns_one_hot = ['country','htap_region', 'climatic_zone', 'type', 'type_of_area']
columns_circular = ['lon']
columns_output = ['o3_average_values', 'o3_daytime_avg', 'o3_nighttime_avg', 'o3_median', 'o3_perc25', 'o3_perc75', 'o3_perc90', 'o3_perc98', 'o3_dma8eu', 'o3_avgdma8epax', 'o3_drmdmax1h', 'o3_w90', 'o3_aot40', 'o3_nvgt070', 'o3_nvgt100']

placeholder_missing_value = -999

def rmv_missing_values(data):
    df_nan = data.replace(-999, np.nan)
    rows_nan = df_nan.isnull().any(axis=1)
    print("-- Missing values detected, rows affected:", rows_nan.sum(), "--")
    df_clean = df_nan.dropna()
    print("-- Missing values deleted, rows left:", df_clean.shape[0] , "--")
    return df_clean

def standardize_values(data):
    data_cont = data.drop(columns=columns_one_hot)
    data_cont = data_cont.drop(columns=['dataset', 'id'])
    print("-- Standardize continous values, columns affected:", data_cont.shape[1] ,"--")
    std_scaler = StandardScaler()
    data_std = pd.DataFrame(std_scaler.fit_transform(data_cont), columns=data_cont.columns)
    data_std = data_std.reset_index(drop=True)
    data_one_hot = data[columns_one_hot].reset_index(drop=True)
    data_dataset_id = data[['dataset', 'id']].reset_index(drop=True)
    data_std_all = pd.concat([data_std, data_one_hot],axis=1)
    data_std_all = pd.concat([data_std_all, data_dataset_id],axis=1)
    return data_std_all
    
def one_hot_encode_values(data):
    print("-- one-hot encode categorical values, columns affected:" ,"--")
    data_to_one_hot = data[columns_one_hot]
    data_rest = data.drop(columns=columns_one_hot)
    data_one_hot = pd.get_dummies(data_to_one_hot, columns=columns_one_hot).astype(int)
    data_one_hot = data_one_hot.reset_index(drop=True)
    data_rest = data_rest.reset_index(drop=True)
    data_all = pd.concat([data_rest, data_one_hot],axis=1)
    return data_all

def split_data_and_cleanup(data):
    train = data[data['dataset'] == 'train']
    train = train.drop(columns=['dataset', 'id'])
    val = data[data['dataset'] == 'val']
    val = val.drop(columns=['dataset', 'id'])
    test = data[data['dataset'] == 'test']
    test = test.drop(columns=['dataset', 'id'])
    return train, val, test

def process(data):
    data_clean = rmv_missing_values(data)
    data_process_one = data_clean.drop(columns=columns_circular)
    data_process_two = standardize_values(data_process_one)
    data_process_three = one_hot_encode_values(data_process_two)
    input_data = data_process_three.drop(columns=columns_output)
    output_data = data_process_three[columns_output + ['dataset', 'id']]
    X_train, X_val, X_test = split_data_and_cleanup(input_data)
    y_train, y_val, y_test = split_data_and_cleanup(output_data)
    return X_train, X_val, X_test, y_train, y_val, y_test
    