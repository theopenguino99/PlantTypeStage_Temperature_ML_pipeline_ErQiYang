"""
Module for cleaning and preprocessing data.
"""
import os
from pathlib import Path
from data_loader import DataLoader
import pandas as pd
import numpy as np
from loguru import logger
import load_preprocessing_config


class DataCleaner:
    """Class to clean and preprocess data."""
    
    def __init__(self, config):
        """
        Initialize the DataCleaner.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = self._load_preprocessing_config()
        
    def _load_preprocessing_config(self):
        """Load preprocessing configuration."""
        import yaml
        dir = os.path.join(Path(__file__).resolve().parents[2], self.config['pipeline']['preprocessing_config'])
        with open(dir, 'r') as file:
            return yaml.safe_load(file)
        
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
        df_cleaned = self._handle_duplicates(df_cleaned)
        df_cleaned = self._handle_missing_values(df_cleaned)
        df_cleaned = self._clean_categorical_variables(df_cleaned)
        df_cleaned = self._clean_numerical_variables(df_cleaned)
        df_cleaned = self._clean_datetime_variables(df_cleaned)
        df_cleaned = self._handle_outliers(df_cleaned)
        
        logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
        
        return df_cleaned
    
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
        logger.info("Handling missing values")
        
        # Get missing value configurations
        num_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['numerical']['strategy']
        cat_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['categorical']['strategy']
        dt_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['datetime']['strategy']
        
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
        
        # Handle missing datetime values
        for col in dt_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                if dt_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif dt_strategy == 'constant':
                    const_val = self.preprocessing_config['cleaning']['handle_missing_values']['datetime']['constant_value']
                    df[col] = df[col].fillna(const_val)
        
        # Report missing values after imputation
        missing_info_after = df.isnull().sum()
        if missing_info_after.sum() > 0:
            logger.warning(f"Missing values after imputation:\n{missing_info_after[missing_info_after > 0]}")
        
        return df
    
    def _clean_categorical_variables(self, df):
        """Clean and standardize categorical variables."""
        logger.info("Cleaning categorical variables")
        
        # Standardize categorical variables specified in config
        if self.preprocessing_config['feature_engineering']['standardize_categorical']['enabled']:
            for col, settings in self.preprocessing_config['feature_engineering']['standardize_categorical']['columns'].items():
                if col in df.columns:
                    # Apply mapping if defined
                    if 'mapping' in settings:
                        logger.info(f"Applying mapping to {col}")
                        df[col] = df[col].map(settings['mapping']).fillna(df[col])
                    
                    # Convert to lowercase if specified
                    if 'to_lowercase' in settings and settings['to_lowercase'] and df[col].dtype == 'object':
                        logger.info(f"Converting {col} to lowercase")
                        df[col] = df[col].str.lower()
        
        return df
    
    def _clean_numerical_variables(self, df):
        """Clean numerical variables, such as age in this case."""
        logger.info("Cleaning numerical variables")
        
        # In this example, we'll handle the age column as mentioned in the requirements
        if 'age' in df.columns:
            logger.info("Cleaning age column")
            
            # Identify invalid age values (not 15 or 16)
            mask = ~df['age'].isin([15, 16])
            invalid_age_count = mask.sum()
            
            if invalid_age_count > 0:
                logger.warning(f"Found {invalid_age_count} invalid age values. Replacing with median age.")
                
                # Replace invalid ages with median of valid ages
                valid_ages = df.loc[~mask, 'age']
                median_age = valid_ages.median() if not valid_ages.empty else 15.5
                df.loc[mask, 'age'] = median_age
        
        return df
    
    def _clean_datetime_variables(self, df):
        """Clean and convert datetime variables."""
        logger.info("Cleaning datetime variables")
        
        datetime_cols = self.preprocessing_config['columns']['datetime']
        
        for col in datetime_cols:
            if col in df.columns:
                # Convert to proper datetime format if it's not already
                if df[col].dtype == 'object':
                    logger.info(f"Converting {col} to proper datetime format")
                    try:
                        # For sleep_time and wake_time, we'll assume they're in HH:MM format
                        df[col] = pd.to_datetime(df[col], format='%H:%M', errors='coerce')
                    except Exception as e:
                        logger.error(f"Error converting {col} to datetime: {e}")
                
                # Handle missing datetime values
                if df[col].isnull().sum() > 0:
                    dt_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['datetime']['strategy']
                    if dt_strategy == 'mode':
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif dt_strategy == 'constant':
                        const_val = self.preprocessing_config['cleaning']['handle_missing_values']['datetime']['constant_value']
                        df[col] = df[col].fillna(const_val)
        
        return df
    
    def _handle_outliers(self, df):
        """Handle outliers in numerical features based on the specified strategy."""
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
    
# To test module
df = DataLoader({'paths': {'raw_data': 'data/raw/score.db'}}).load_data()
cleaner = DataCleaner({'pipeline': {'preprocessing_config': 'config/preprocessing_config.yaml'}})
cleaned_data = cleaner.clean_data(df)
print(cleaned_data.head())