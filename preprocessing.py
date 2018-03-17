import tensorflow as tf
import pandas as pd
import numpy as np
from helper import get_time_from_datetime, get_dummy_values_for_quant_features, get_scaled_features, get_test_train_data ,get_target_features

def preprocessing():
   rides = pd.read_csv('train.csv');
   ##Get time from datetime
   get_time_from_datetime(rides)
   ## Dummy varibales for categorical columns
   categorcal_columns = ["season", "holiday", "workingday", "weather"]
   rides = get_dummy_values_for_quant_features(categorcal_columns, rides)
   ## Scaling quant features in the data
   quant_columns = ["temp", "atemp", "humidity", "windspeed"]
   rides = get_scaled_features(quant_columns, rides)
   ## Drop columns atemp
   rides = rides.drop(["atemp"], axis=1)
   train_data, test_data = get_test_train_data(10, rides)
   target_features = ["casual", "registered", "count"]
   train_data_features, train_data_targets = get_target_features(target_features, train_data)
   test_data_features, test_data_targets = get_target_features(target_features, test_data)
   return train_data_features.iloc[1:], train_data_targets.iloc[1:], test_data_features.iloc[1:], test_data_targets.iloc[1:]