import pytest
import pandas as pd
from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_selection import FeatureSelector
from feature_engineering import FeatureEngineer
from data_preprocessor import DataPreprocessor

# Load data
data_loader = DataLoader()
data = data_loader.load_data()

# Show BEFORE
print(data.info())

# Clean data
data_cleaner = DataCleaner()
data_cleaned = data_cleaner.clean_data(data)
# Feature selection
feature_selector = FeatureSelector()
data_selected = feature_selector.select_features(data_cleaned)
# Feature engineering
feature_engineer = FeatureEngineer()
data_engineered = feature_engineer.engineer_features(data_selected)
# Preprocess data
preprocessor = DataPreprocessor()   
data_preprocessed = preprocessor.preprocess(data_engineered)

# Show AFTER
print(data_preprocessed.info())