"""
Module for feature engineering.
"""

import pandas as pd
import numpy as np
from loguru import logger
from config_loader import *

class FeatureEngineer:
    """Class to engineer new features from existing data."""
    
    def __init__(self):
        """
        Initialize the FeatureEngineer.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
    
    def engineer_features(self, df):
        """
        Apply feature engineering transformations to the dataset.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        df_features = df.copy()

        # Apply each feature engineering step defined in the config
        if self.config['create_nutrient_balance_index']['enabled']:
            df_features = self._create_nutrient_balance_index(df_features)
        
        if self.config['create_environmental_stress_index']['enabled']:
            df_features = self._create_environmental_stress_index(df_features)
        
        if self.config['create_water_quality_score']['enabled']:
            df_features = self._create_water_quality_score(df_features)
        
        if self.config['standardize_categorical']['enabled']:
            df_features = self._standardize_categorical(df_features)

        return df_features

    def _create_nutrient_balance_index(self, df):
        """Create nutrient balance index from N, P, and K sensors."""
        required_cols = self.config['create_nutrient_balance_index']['from_columns']
        if all(col in df.columns for col in required_cols):
            df['nutrient_balance_index'] = (
                df[required_cols[0]].astype(float) +
                df[required_cols[1]].astype(float) +
                df[required_cols[2]].astype(float)
            ) / 3
        return df

    def _create_environmental_stress_index(self, df):
        """Create environmental stress index using multiple environmental sensors."""
        required_cols = self.config['create_environmental_stress_index']['from_columns']
        if all(col in df.columns for col in required_cols):
            df['environmental_stress_index'] = (
                df[required_cols[0]] +
                df[required_cols[1]] +
                df[required_cols[2]] +
                df[required_cols[3]] +
                df[required_cols[4]]
            ) / 5
        return df

    def _create_water_quality_score(self, df):
        """Create water quality score based on pH, EC, and water level sensors."""
        required_cols = self.config['create_water_quality_score']['from_columns']
        if all(col in df.columns for col in required_cols):
            df['water_quality_score'] = (
                df[required_cols[0]] +
                df[required_cols[1]] +
                df[required_cols[2]]
            ) / 3
        return df

    def _standardize_categorical(self, df):
        """Standardize categorical columns by converting them to lowercase."""
        for col in self.config['standardize_categorical']['columns']:
            if col in df.columns:
                df[col] = df[col].str.lower()
        return df
# Test the feature engineering module
if __name__ == "__main__":
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    data_loader = DataLoader()
    df = data_loader.load_data()
    data_engineer = FeatureEngineer()
    df_features = data_engineer.engineer_features(df)
    print(df_features.head())