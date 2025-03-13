"""
Module for cleaning and preprocessing data.
"""
import os
from pathlib import Path
from data_loader import DataLoader
import pandas as pd
import numpy as np
from loguru import logger
from config_loader import load_config, load_preprocessing_config


class DataCleaner:
    """Class to clean and preprocess data."""
    
    def __init__(self):
        """
        Initialize the DataCleaner.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
        
    def clean_data(self, df):
        """
        Clean and preprocess the data.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning process")
        
        # Create a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Apply each cleaning step
        df_cleaned = self.clean_nutrient_sensors(df_cleaned)
        df_cleaned = self._map_labels_to_lowercase(df_cleaned)
        df_cleaned = self._handle_negative_values(df_cleaned)
        df_cleaned = self._handle_duplicates(df_cleaned)
        df_cleaned = self._handle_missing_values(df_cleaned)
        df_cleaned = self._handle_outliers(df_cleaned)
        
        logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
        
        return df_cleaned
    def clean_nutrient_sensors(self, df):
        """
        Clean and extract numerical values from nutrient sensor columns.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with cleaned nutrient sensor columns
        """
        if not self.preprocessing_config['cleaning']['clean_Nutrient_Sensor']['enabled']:
            return df
        
        logger.info("Cleaning nutrient sensor columns")
        
        nutrient_columns = self.preprocessing_config['cleaning']['clean_Nutrient_Sensor']['columns']
        
        for col in nutrient_columns:
            if col in df.columns:
                logger.info(f"Extracting numerical values from {col}")
                df[col] = df[col].str.extract('(\d+)', expand=False).astype(float)
        
        return df
        
    def _map_labels_to_lowercase(self, df):
        """Map capitalized labels in specified columns to lowercase."""
        if not self.preprocessing_config['cleaning']['map_labels_to_lowercase']['enabled']:
            return df
        
        logger.info("Mapping labels to lowercase")
        
        columns_to_map = self.preprocessing_config['cleaning']['map_labels_to_lowercase']['columns']
        
        for col in columns_to_map:
            if col in df.columns:
                logger.info(f"Mapping labels in {col} to lowercase")
                df[col] = df[col].str.lower()
        
        return df
    
    def _handle_negative_values(self, df):
        """Handle negative values in specified columns."""
        if not self.preprocessing_config['cleaning']['handle_negative_values']['enabled']:
            return df
        
        logger.info("Handling negative values in specified columns")
        
        columns_to_handle = self.preprocessing_config['cleaning']['handle_negative_values']['columns']
        strategy = self.preprocessing_config['cleaning']['handle_negative_values']['strategy']
        
        for col in columns_to_handle:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logger.info(f"Found {negative_count} negative values in {col}")
                    if strategy == 'remove':
                        df = df[df[col] >= 0]
                        logger.info(f"Removed rows with negative values in {col}")
                    elif strategy == 'absolute':
                        df[col] = df[col].abs()
                        logger.info(f"Converted negative values to absolute in {col}")
        
        return df
    
    def _handle_duplicates(self, df):
        """Handle duplicate rows in the dataframe."""
        if not self.preprocessing_config['cleaning']['handle_duplicates']['enabled']:
            return df
        
        logger.info("Handling duplicate rows")
        
        # Identify columns to ignore when detecting duplicates
        ignore_cols = self.preprocessing_config['cleaning']['handle_duplicates']['ignore_columns']
        cols_to_check = [col for col in df.columns if col not in ignore_cols]
        
        # Count duplicates
        n_duplicates = df.duplicated(subset=cols_to_check).sum()
        logger.info(f"Found {n_duplicates} duplicate rows")
        
        if n_duplicates > 0:
            # Keep only the first occurrence of duplicates
            keep_option = self.preprocessing_config['cleaning']['handle_duplicates']['keep']
            df = df.drop_duplicates(subset=cols_to_check, keep=keep_option)
            logger.info(f"Removed duplicates. Keeping '{keep_option}' occurrences.")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataframe."""
        if not self.preprocessing_config['cleaning']['handle_missing_values']['enabled']:
            return df
        logger.info("Handling missing values")
        
        # Get missing value configurations
        num_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['numerical']['strategy']
        cat_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['categorical']['strategy']
        
        # Get column lists
        num_cols = self.preprocessing_config['columns']['numerical']
        cat_cols = self.preprocessing_config['columns']['categorical']
        dt_cols = self.preprocessing_config['columns']['datetime']
        
        # Report missing values
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            logger.info(f"Missing values before imputation:\n{missing_info[missing_info > 0]}")
        
        # Handle missing numerical values
        for col in num_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                if num_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif num_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif num_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif num_strategy == 'constant':
                    const_val = self.preprocessing_config['cleaning']['handle_missing_values']['numerical']['constant_value']
                    df[col] = df[col].fillna(const_val)
        
        # Handle missing categorical values
        for col in cat_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                if cat_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif cat_strategy == 'constant':
                    const_val = self.preprocessing_config['cleaning']['handle_missing_values']['categorical']['constant_value']
                    df[col] = df[col].fillna(const_val)
        
        # Report missing values after imputation
        missing_info_after = df.isnull().sum()
        if missing_info_after.sum() > 0:
            logger.warning(f"Missing values after imputation:\n{missing_info_after[missing_info_after > 0]}")
        
        return df
    
    
    def _handle_outliers(self, df):
        """Handle outliers in numerical features based on the specified strategy."""
        if not self.preprocessing_config['cleaning']['handle_outliers']['enabled']:
            return df
        logger.info("Handling outliers")
        
        if self.preprocessing_config['cleaning']['handle_outliers']==None:
            return df
        
        method = self.preprocessing_config['cleaning']['handle_outliers']['method']
        iqr_multiplier = self.preprocessing_config['cleaning']['handle_outliers'].get('iqr_multiplier', 1.5)
        zscore_threshold = self.preprocessing_config['cleaning']['handle_outliers'].get('zscore_threshold', 3.0)
        outlier_columns = self.preprocessing_config['cleaning']['handle_outliers'].get('columns', [])
        
        for col in outlier_columns:
            if col in df.columns:
                if method == 'zscore':
                    from scipy.stats import zscore
                    z_scores = zscore(df[col].dropna())
                    outliers = abs(z_scores) > zscore_threshold
                    df.loc[outliers, col] = df[col].median()
                    logger.info(f"Replaced {outliers.sum()} outliers in {col} using Z-score method.")
                
                elif method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                    df.loc[outliers, col] = df[col].median()
                    logger.info(f"Replaced {outliers.sum()} outliers in {col} using IQR method.")
        
        return df
    
# # To test module
# df = DataLoader().load_data()
# cleaner = DataCleaner()
# cleaned_data = cleaner.clean_data(df)
# print(cleaned_data.head())
# print(cleaned_data.info())