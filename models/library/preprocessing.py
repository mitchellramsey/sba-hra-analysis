from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
import pandas as pd

def create_train_test_split(df, target, ts=0.3, rs=42):
    """
        Creates the train_test_split for the provided dataframe. Function must specify target feature. Allows option
        to pass in own randome_state and test_size. If none is provided, 42 and 0.3 will be utilized, respectively. 
    """        
    X = df.drop(columns=target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=ts, random_state=rs)

def standard_scale_X_datasets(X_train, X_test):
    """
        Takes the respective X_train and X_test data splits and scales them with the Standard Scaler. The scaler is
        fitted with the X_train dataset and both datasets are transformed. Returns both transformed datasets.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def robust_scale_X_datasets(X_train, X_test):
    """
       Takes the respective X_train and X_test data splits and scales them with the Robust Scaler. The scaler is
        fitted with the X_train dataset and both datasets are transformed. Returns both transformed datasets. 
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_rb = scaler.transform(X_train)
    X_test_rb = scaler.transform(X_test)
    return X_train_rb, X_test_rb

def processing_pipeline(df, target_col=None, scaler='Standard', ):
  if target_col == None:
    raise Exception('The target feature must be specified')
  X_train, X_test, y_train, y_test = create_train_test_split(df, target=target_col)
  if scaler == 'Standard':
    X_train_scaled, X_test_scaled = standard_scale_X_datasets(X_train, X_test)
  else:
    X_train_scaled, X_test_scaled = robust_scale_X_datasets(X_train, X_test)
  
  return X_train_scaled, X_test_scaled, y_train, y_test