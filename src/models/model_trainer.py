"""
Module for training machine learning models.
"""

import os
import joblib
import pandas as pd
import numpy as np
import yaml
from loguru import logger
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold
import optuna

class ModelTrainer:
    """Class to train machine learning models."""
    
    def __init__(self, config):
        """
        Initialize the ModelTrainer.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.model_config = self._load_model_config()
        self.models = {}
        self.best_params = {}
        
    def _load_model_config(self):
        """Load model configuration."""
        with open(self.config['pipeline']['model_config'], 'r') as file:
            return yaml.safe_load(file)
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple models specified in the configuration.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            X_val (pandas.DataFrame): Validation features
            y_val (pandas.Series): Validation target
            
        Returns:
            dict: Dictionary of trained models
        """
        logger.info("Starting model training")
        
        # Initialize dictionary to store trained models
        trained_models = {}
        
        # Get list of models to train from config
        models_to_train = {k: v for k, v in self.model_config['models'].items() if v['enabled']}
        
        # Train each model
        for model_name, model_config in models_to_train.items():
            logger.info(f"Training {model_name}")
            
            # Initialize model with default parameters
            model = self._initialize_model(model_name, model_config['params'])
            
            # Check if hyperparameter tuning is enabled
            if model_config['hyperparameter_tuning']['enabled']:
                logger.info(f"Performing hyperparameter tuning for {model_name}")
                model = self._tune_hyperparameters(
                    model_name, model, model_config['hyperparameter_tuning']['param_grid'], 
                    X_train, y_train, X_val, y_val
                )
            else:
                # Fit model with default parameters
                model.fit(X_train, y_train)
            
            # Store trained model
            trained_models[model_name] = model
            self.models[model_name] = model
            
            # Save model to disk
            if self.config['experiment']['save_models']:
                self._save_model(model, model_name)
        
        logger.info(f"Model training completed. Trained {len(trained_models)} models")
        return trained_models
    
    def _initialize_model(self, model_name, params):
        """Initialize a model with given parameters."""
        # Get random state from common settings
        random_state = self.model_config['common']['random_state']
        n_jobs = self.model_config['common']['n_jobs']
        
        # Add random_state to params if model supports it
        if model_name not in ['linear_regression']:
            params['random_state'] = random_state
        
        # Add n_jobs to params if model supports it
        if model_name in ['random_forest', 'xgboost', 'lightgbm']:
            params['n_jobs'] = n_jobs
        
        # Initialize model based on name
        if model_name == 'linear_regression':
            return LinearRegression(**params)
            
        elif model_name == 'ridge_regression':
            return Ridge(**params)
            
        elif model_name == 'lasso_regression':
            return Lasso(**params)
            
        elif model_name == 'elastic_net':
            return ElasticNet(**params)
            
        elif model_name == 'random_forest':
            return RandomForestRegressor(**params)
            
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(**params)
            
        elif model_name == 'xgboost':
            return xgb.XGBRegressor(**params)
            
        elif model_name == 'lightgbm':
            return lgb.LGBMRegressor(**params)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _tune_hyperparameters(self, model_name, model, param_grid, X_train, y_train, X_val, y_val):
        """
        Tune hyperparameters for a model.
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            param_grid (dict): Parameter grid for hyperparameter tuning
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Model with best parameters
        """
        # Use cross-validation if specified
        if self.config['pipeline']['use_cross_validation']:
            n_folds = self.config['pipeline']['n_folds']
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.model_config['common']['random_state'])
            
            # Check if we should use GridSearchCV or Optuna
            if isinstance(param_grid, dict) and len(param_grid) <= 5:
                # Use GridSearchCV for small parameter spaces
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv, scoring='neg_mean_squared_error', 
                    n_jobs=self.model_config['common']['n_jobs']
                )
                grid_search.fit(X_train, y_train)
                
                best_params = grid_search.best_params_
                best_model = grid_search.best_estimator_
                
            else:
                # Use Optuna for larger parameter spaces
                best_params, best_model = self._optuna_tuning(
                    model_name, model, param_grid, X_train, y_train, cv
                )
                
        else:
            # Use validation set directly
            best_params, best_model = self._optuna_tuning(
                model_name, model, param_grid, X_train, y_train, None, X_val, y_val
            )
        
        # Store best parameters
        self.best_params[model_name] = best_params
        logger.info(f"Best parameters for {model_name}: {best_params}")
        
        return best_model
    
    def _optuna_tuning(self, model_name, model, param_grid, X_train, y_train, cv=None, X_val=None, y_val=None):
        """
        Tune hyperparameters using Optuna.
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            param_grid (dict): Parameter grid for hyperparameter tuning
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            cv (KFold, optional): Cross-validation strategy
            X_val (pandas.DataFrame, optional): Validation features
            y_val (pandas.Series, optional): Validation target
            
        Returns:
            tuple: Best parameters and best model
        """
        def objective(trial):
            # Sample hyperparameters from the parameter grid
            params = {key: trial.suggest_categorical(key, value) if isinstance(value, list) else trial.suggest_float(key, *value) for key, value in param_grid.items()}
            model.set_params(**params)
            
            # Perform cross-validation or use validation set
            if cv:
                scores = []
                for train_idx, val_idx in cv.split(X_train):
                    X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    model.fit(X_t, y_t)
                    preds = model.predict(X_v)
                    score = np.mean((preds - y_v) ** 2)
                    scores.append(score)
                return np.mean(scores)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return np.mean((preds - y_val) ** 2)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config['pipeline']['optuna_trials'])
        
        best_params = study.best_params
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        
        return best_params, model