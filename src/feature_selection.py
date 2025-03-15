"""
Module for feature selection.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from config_loader import *

class FeatureSelector:
    """Class to select features for model training."""
    
    def __init__(self, problem_type):
        """
        Initialize the FeatureSelector.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.selected_features = None
        self.preprocessing_config = load_preprocessing_config()
        self.problem_type = problem_type
        if self.problem_type == 'regression':
            target_column = self.config['data']['target_num']
        elif self.problem_type == 'classification':
            target_column = self.config['data']['target_cat']
        else:
            raise ValueError(f"Unsupported problem type '{self.problem_type}'")
        self.target_column = target_column
    
    
    def select_features(self, df):
        """
        Select features for model training and split data.
        
        Args:
            df (pandas.DataFrame): Input dataframe with engineered features
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """

        
        # Copy dataframe to avoid modifying the original
        df_select = df.copy()

        # Create new column that concatinates features "Plant Type" and "Plant Stage" to create "Plant Type-Stage"
        df_select['Plant Type-Stage'] = df_select['Plant Type'] + '-' + df_select['Plant Stage']
        
        # Delete original columns if enabled in configuration
        if self.preprocessing_config['feature_engineering']['delete_Type_Stage_AfterCombination']['enabled']:
            df_select.drop(['Plant Type', 'Plant Stage'], axis=1, inplace=True)
        
        # Get the feature selection method from config
        method = self.preprocessing_config['feature_selection']['method'].lower()
        
        # Apply feature selection if specified
        if method != 'none':
            
            # Extract target variable before feature selection
            X = df_select.drop(columns=[self.target_column])
            y = df_select[self.target_column]
            
            # Select features based on the specified method
            if method == 'variance':
                X = self._select_by_variance(X)
            elif method == 'correlation':
                X = self._select_by_correlation(X, y)
            elif method == 'importance':
                X = self._select_by_importance(X, y)
            
            # Combine features and target for splitting
            df_select = pd.concat([X, y], axis=1)
        
        return df_select
    
    def _select_by_variance(self, X):
        """Select features based on variance threshold."""
        threshold = self.preprocessing_config['feature_selection']['variance_threshold']
        selector = VarianceThreshold(threshold)
        
        # Fit the selector
        X_selected = selector.fit_transform(X)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self.selected_features = selected_features
        
        return X[selected_features]
    
    def _select_by_correlation(self, X, y):
        """Select features based on correlation with target."""
        threshold = self.preprocessing_config['feature_selection']['correlation_threshold']
        
        # Calculate correlation with target
        X_with_target = pd.concat([X, y], axis=1)
        correlation = X_with_target.corr()[self.target_column].abs().sort_values(ascending=False)
        
        # Select features with correlation above threshold
        selected_features = correlation[correlation > threshold].index.tolist()
        # Remove target from selected features
        if self.target_column in selected_features:
            selected_features.remove(self.target_column)
        
        self.selected_features = selected_features
        
        return X[selected_features]
    
    def _select_by_importance(self, X, y):
        """Select features based on feature importance from a model."""
        # Use Random Forest for feature importance
        from sklearn.ensemble import RandomForestRegressor
        
        # Train a Random Forest with limited depth for quick feature importance
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=self.config['data']['random_state']
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top 80% of cumulative importance
        cumulative_importance = np.cumsum(importances[indices])
        importance_threshold = 0.8  # 80% of cumulative importance
        
        # Select features based on cumulative importance
        n_features = np.where(cumulative_importance >= importance_threshold)[0][0] + 1
        selected_indices = indices[:n_features]
        selected_features = X.columns[selected_indices].tolist()
        
        self.selected_features = selected_features
        
        return X[selected_features]
    
    def _split_data(self, df):
        """Split data into train, validation, and test sets."""

        # Extract target variable
        target_column = self.target_column
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
        
        return X_train, X_val, X_test, y_train, y_val, y_test