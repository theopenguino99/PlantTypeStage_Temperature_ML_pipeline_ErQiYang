#!/usr/bin/env python3
"""
Main module to orchestrate processing of datasets according to classification or regression problem
"""

from loguru import logger
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector

# Define class to load data according to regression or classification problem:
class DataInput:
    def __init__(self, problem_type = None):

        self.problem_type = problem_type
    
    def process_data(self):

        # Check problem type
        if self.problem_type not in ['classification', 'regression']:
            raise ValueError(f"Unsupported problem type '{self.problem_type}. \nPlease define either 'classification' or 'regression'")

        # Load->Clean->Select->Engineer->Preprocess (Select and Preprocess depends on problem type)
        logger.info("Loading data")
        data = DataLoader().load_data()
        logger.info("Cleaning data")
        cleaned_data = DataCleaner().clean_data(data)
        logger.info("Selecting features")
        data_selected = FeatureSelector(self.problem_type).select_features(cleaned_data)
        logger.info("Engineering features")
        data_engineered = FeatureEngineer().engineer_features(data_selected)
        logger.info("Preprocessing data")
        preprocessed_data = DataPreprocessor(self.problem_type).preprocess(data_engineered)
        
        return preprocessed_data
    
# # Test the module
# import pandas as pd
# pd.set_option('display.max_columns', None)
# if __name__ == "__main__":
#     # Load data for regression problem
#     logger.info("Loading data for regression problem")
#     regression_data = DataInput('regression').process_data()
#     logger.info(f"Regression data: {regression_data.head()}")
    
#     # Load data for classification problem
#     logger.info("Loading data for classification problem")
#     classification_data = DataInput('classification').process_data()
#     logger.info(f"Classification data: {classification_data.head()}")
