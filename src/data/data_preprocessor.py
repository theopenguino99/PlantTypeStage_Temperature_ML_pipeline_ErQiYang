"""
Module for preprocessing data before model training.
"""

import yaml
from pathlib import Path
import os
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder, TargetEncoder
from datetime import datetime
from loguru import logger
import load_preprocessing_config

class DataPreprocessor:
    """Class to preprocess data for model training."""
    
    def __init__(self, config):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = load_preprocessing_config(self.config) # ERROR HERERERE ?????
        self.encoders = {}
        self.scalers = {}
    
    
    def preprocess(self, df):
        """
        Preprocess the data for model training.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Preprocessed dataframe
        """
        logger.info("Starting data preprocessing")
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Drop unnecessary columns
        df_processed = self._drop_columns(df_processed)
        
        # Process datetime columns
        df_processed = self._process_datetime_columns(df_processed)
        
        # Encode categorical variables
        df_processed = self._encode_categorical_columns(df_processed)
        
        # Scale numerical variables
        df_processed = self._scale_numerical_columns(df_processed)
        
        # Save processed data
        print(self.config['paths']['processed_data'])
        dir = os.path.join(Path(__file__).resolve().parents[2], self.config['paths']['processed_data'])
        df_processed.to_csv(dir, index=False)
        logger.info(f"Preprocessed data saved to {dir}")
        
        return df_processed
    
    def _drop_columns(self, df):
        """Drop unnecessary columns."""
        drop_columns = self.preprocessing_config['columns']['drop_columns']
        if drop_columns:
            # Only drop columns that exist in the dataframe
            columns_to_drop = [col for col in drop_columns if col in df.columns]
            if columns_to_drop:
                logger.info(f"Dropping columns: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop)
        return df
    
    def _process_datetime_columns(self, df):
        """Process datetime columns to extract features."""
        datetime_cols = self.preprocessing_config['columns']['datetime']
        datetime_features = self.preprocessing_config['preprocessing']['datetime_features']
        
        for col in datetime_cols:
            if col not in df.columns:
                continue
                
            logger.info(f"Processing datetime column: {col}")
            
            # Convert to datetime if needed
            if df[col].dtype == 'object':
                try:
                    # For time columns like sleep_time, wake_time
                    df[col] = pd.to_datetime(df[col], format='%H:%M', errors='coerce')
                except:
                    # Try general datetime parsing
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Extract features if specified in config
            col_prefix = f"{col}_"
            
            if datetime_features['extract_hour']:
                df[f"{col_prefix}hour"] = df[col].dt.hour
                
            if datetime_features['extract_minute']:
                df[f"{col_prefix}minute"] = df[col].dt.minute
        
        # Calculate time difference between sleep_time and wake_time
        if (datetime_features['extract_time_difference'] and 
            'sleep_time' in df.columns and 'wake_time' in df.columns):
            
            logger.info("Calculating sleep duration")
            
            # Convert times to total minutes for calculation
            sleep_minutes = df['sleep_time'].dt.hour * 60 + df['sleep_time'].dt.minute
            wake_minutes = df['wake_time'].dt.hour * 60 + df['wake_time'].dt.minute
            
            # Adjust for overnight sleep (when sleep_time > wake_time)
            sleep_duration = np.where(
                sleep_minutes > wake_minutes,
                (24 * 60 - sleep_minutes) + wake_minutes,  # Overnight sleep
                wake_minutes - sleep_minutes  # Same-day sleep
            )
            
            df['sleep_duration'] = sleep_duration
            
            # Drop original datetime columns after extracting features
            df = df.drop(columns=datetime_cols)
        
        return df
    
    def _encode_categorical_columns(self, df):
        """Encode categorical variables."""
        categorical_cols = self.preprocessing_config['columns']['categorical']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if not categorical_cols:
            return df
            
        encoding_method = self.preprocessing_config['preprocessing']['categorical_encoding']['method']
        handle_unknown = self.preprocessing_config['preprocessing']['categorical_encoding']['handle_unknown']
        
        logger.info(f"Encoding categorical variables using {encoding_method} encoding")
        
        if encoding_method == 'one-hot':
            encoder = OneHotEncoder(cols=categorical_cols, handle_unknown=handle_unknown)
            df = encoder.fit_transform(df)
            self.encoders['categorical'] = encoder
            
        elif encoding_method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                
        elif encoding_method == 'target':
            # Will be implemented in the feature engineering step
            # as it requires the target variable
            pass
            
        elif encoding_method == 'frequency':
            for col in categorical_cols:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map)
                self.encoders[col] = freq_map
        
        return df
    
    def _scale_numerical_columns(self, df):
        """Scale numerical variables."""
        numerical_cols = self.preprocessing_config['columns']['numerical']
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if not numerical_cols:
            return df
            
        scaling_method = self.preprocessing_config['preprocessing']['numerical_scaling']['method']
        
        if scaling_method == 'none':
            return df
            
        logger.info(f"Scaling numerical variables using {scaling_method} scaling")
        
        # Create a copy of the numerical columns for scaling
        df_scaled = df.copy()
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
            
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        
        # Fit and transform the numerical columns
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.scalers['numerical'] = scaler
        
        return df_scaled
    
    def split_data(self, df):
        """
        Split data into training, validation, and test sets.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # Extract target variable
        target_column = self.config['data']['target_column']
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split into train+val and test
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split train+val into train and validation
        val_size = self.config['data']['validation_size']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# To test module
if __name__ == "__main__":
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    config_path = os.path.join(Path(__file__).resolve().parents[2], 'config/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    data_loader = DataLoader(config)
    data = data_loader.load_data()
    data_cleaner = DataCleaner(config)
    cleaned_data = data_cleaner.clean_data(data)
    preprocessor = DataPreprocessor(config)
    preprocessed_data = preprocessor.preprocess(cleaned_data)
    preprocessed_data.head()