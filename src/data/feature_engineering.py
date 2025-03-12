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
        Engineer new features from the data.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Apply each feature engineering step defined in the config
        feature_steps = self.preprocessing_config['feature_engineering']
        
        # Create sleep duration (if not already created in preprocessing)
        if feature_steps['create_sleep_duration']['enabled']:
            df_features = self._create_sleep_duration(df_features)
        
        # Create class size
        if feature_steps['create_class_size']['enabled']:
            df_features = self._create_class_size(df_features)
        
        # Create gender ratio
        if feature_steps['create_gender_ratio']['enabled']:
            df_features = self._create_gender_ratio(df_features)
        
        # Create study efficiency
        if feature_steps['create_study_efficiency']['enabled']:
            df_features = self._create_study_efficiency(df_features)
        
        # Create additional features based on domain knowledge
        df_features = self._create_additional_features(df_features)
        
        # Save engineered features data
        features_data_path = self.config['paths']['features_data']
        df_features.to_csv(features_data_path, index=False)
        logger.info(f"Engineered features saved to {features_data_path}")
        
        return df_features
    
    def _create_sleep_duration(self, df):
        """Create sleep duration feature if sleep_time and wake_time are in the dataframe."""
        required_cols = self.preprocessing_config['feature_engineering']['create_sleep_duration']['from_columns']
        
        # Check if sleep_duration was already created in preprocessing
        if 'sleep_duration' in df.columns:
            return df
        
        # Check if required columns exist
        if all(col in df.columns for col in required_cols):
            logger.info("Creating sleep duration feature")
            
            # If columns are already datetime type
            if hasattr(df[required_cols[0]], 'dt'):
                # Convert times to total minutes for calculation
                sleep_minutes = df[required_cols[0]].dt.hour * 60 + df[required_cols[0]].dt.minute
                wake_minutes = df[required_cols[1]].dt.hour * 60 + df[required_cols[1]].dt.minute
                
                # Adjust for overnight sleep (when sleep_time > wake_time)
                sleep_duration = np.where(
                    sleep_minutes > wake_minutes,
                    (24 * 60 - sleep_minutes) + wake_minutes,  # Overnight sleep
                    wake_minutes - sleep_minutes  # Same-day sleep
                )
                
                df['sleep_duration'] = sleep_duration
            else:
                # If we have extracted hour and minute features
                sleep_hour_col = f"{required_cols[0]}_hour"
                sleep_min_col = f"{required_cols[0]}_minute"
                wake_hour_col = f"{required_cols[1]}_hour"
                wake_min_col = f"{required_cols[1]}_minute"
                
                if all(col in df.columns for col in [sleep_hour_col, sleep_min_col, wake_hour_col, wake_min_col]):
                    sleep_minutes = df[sleep_hour_col] * 60 + df[sleep_min_col]
                    wake_minutes = df[wake_hour_col] * 60 + df[wake_min_col]
                    
                    # Adjust for overnight sleep
                    sleep_duration = np.where(
                        sleep_minutes > wake_minutes,
                        (24 * 60 - sleep_minutes) + wake_minutes,
                        wake_minutes - sleep_minutes
                    )
                    
                    df['sleep_duration'] = sleep_duration
        
        return df
    
    def _create_class_size(self, df):
        """Create class size feature from number of male and female students."""
        required_cols = self.preprocessing_config['feature_engineering']['create_class_size']['from_columns']
        
        # Check if required columns exist
        if all(col in df.columns for col in required_cols):
            logger.info("Creating class size feature")
            df['class_size'] = df[required_cols[0]] + df[required_cols[1]]
        
        return df
    
    def _create_gender_ratio(self, df):
        """Create gender ratio feature from number of male and female students."""
        required_cols = self.preprocessing_config['feature_engineering']['create_gender_ratio']['from_columns']
        
        # Check if required columns exist
        if all(col in df.columns for col in required_cols):
            logger.info("Creating gender ratio feature")
            
            # Avoid division by zero
            df['gender_ratio'] = df[required_cols[0]] / df[required_cols[1]].replace(0, np.nan)
            
            # Fill NaN values with 0 (no females) or with median
            median_ratio = df['gender_ratio'].median()
            df['gender_ratio'] = df['gender_ratio'].fillna(median_ratio)
        
        return df
    
    def _create_study_efficiency(self, df):
        """Create study efficiency feature from hours per week and attendance rate."""
        required_cols = self.preprocessing_config['feature_engineering']['create_study_efficiency']['from_columns']
        
        # Check if required columns exist
        if all(col in df.columns for col in required_cols):
            logger.info("Creating study efficiency feature")
            
            # Normalize values
            hours_normalized = df[required_cols[0]] / df[required_cols[0]].max()
            attendance_normalized = df[required_cols[1]] / 100  # Assuming attendance is in percentage
            
            # Create efficiency metric
            df['study_efficiency'] = hours_normalized * attendance_normalized
        
        return df
    
    def _create_additional_features(self, df):
        """Create additional features based on domain knowledge."""
        logger.info("Creating additional features")
        
        # Example: Learning environment score
        if all(col in df.columns for col in ['attendance_rate', 'class_size']):
            # Smaller class size is generally better for learning
            class_size_norm = 1 - (df['class_size'] / df['class_size'].max())
            df['learning_environment_score'] = 0.7 * (df['attendance_rate'] / 100) + 0.3 * class_size_norm
        
        # Example: Study-sleep balance
        if all(col in df.columns for col in ['hours_per_week', 'sleep_duration']):
            sleep_hours = df['sleep_duration'] / 60  # Convert minutes to hours
            study_hours = df['hours_per_week'] / 7  # Average study hours per day
            
            # Ideal balance: Not too much study time at expense of sleep
            df['study_sleep_balance'] = (sleep_hours * study_hours) / (sleep_hours + study_hours)
        
        # Example: Direct admission impact
        if 'direct_admission' in df.columns:
            # Convert to binary if it's not already
            if df['direct_admission'].dtype == 'object':
                df['direct_admission_binary'] = df['direct_admission'].map({'Yes': 1, 'No': 0})
        
        # Example: Interaction features
        if all(col in df.columns for col in ['attendance_rate', 'hours_per_week']):
            df['attendance_study_interaction'] = (df['attendance_rate'] / 100) * df['hours_per_week']
        
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