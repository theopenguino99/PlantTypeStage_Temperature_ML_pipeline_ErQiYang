import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
from loguru import logger
from config_loader import load_config, load_preprocessing_config, load_model_config
from plant_type_stage_classification_models import (
    PlantTypeStageClassifier, EnsemblePlantClassifier, AdaptivePlantClassifier
)
from temperature_regression_models import (
    TemperatureRegressionModel, AdaptiveTemperatureRegressor, DeepTemperatureRegressor
)
from model_evaluator import ModelEvaluator

class ModelTrainer:
    """
    Trains machine learning models for both temperature prediction and plant type-stage classification.
    """
    
    def __init__(self):
        """
        Initialize the model trainer with configuration.
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
        self.model_config = load_model_config()
        self.pipeline_config = self.config['pipeline']
        self.data_config = self.config['data']
        self.models_dir = self.config['paths']['models_dir']
        self.use_cross_validation = self.pipeline_config['use_cross_validation']
        self.n_folds = self.pipeline_config['n_folds']
        self.model_evaluator = ModelEvaluator()
        logger.info("ModelTrainer initialized with configuration")

    def prepare_data(self, features_data, target_column):
        """
        Prepare the data for training and testing.
        
        Args:
            features_data (DataFrame): DataFrame with features
            target_column (str): Name of the target column
            
        Returns:
            Tuple containing training and testing data
        """
        # Extract features and target
        X = features_data.drop(columns=[target_column])
        y = features_data[target_column]
        
        # Split data into training and testing sets
        test_size = self.data_config['test_size']
        random_state = self.data_config['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # If validation set is required
        validation_size = self.data_config['validation_size']
        if validation_size > 0:
            train_size = 1 - validation_size
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=train_size, random_state=random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_temperature_models(self, features_data):
        """
        Train regression models to predict temperature.
        
        Args:
            features_data (DataFrame): DataFrame with features
            target_column (str): Name of the temperature column
            
        Returns:
            Dictionary containing trained models and their performance metrics
        """
        logger.info("Starting temperature prediction model training")
        target_column = self.data_config['target_num']

        # Prepare data
        data_splits = self.prepare_data(features_data, target_column)
        if len(data_splits) == 6:
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        else:
            X_train, X_test, y_train, y_test = data_splits
            X_val, y_val = None, None
        
        # Get enabled models from model_config
        enabled_models = [
            model_name for model_name, model_config in self.model_config['models'].items()
            if model_config['enabled'] == True
        ]
        
        results = {}
        best_metric = float('inf')  # Lower is better for regression metrics like RMSE
        best_model = None
        
        # Train standard models
        for model_name in enabled_models:
            
            logger.info(f"Training {model_name} for temperature prediction")
            
            # Create model
            model_params = self.model_config['models'][model_name].get('params', {})
            model = TemperatureRegressionModel(model_name=model_name, model_params=model_params)
    
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Perform hyperparameter tuning if enabled
            hyperparams_config = self.model_config['models'][model_name].get('hyperparameter_tuning', {})
            if hyperparams_config.get('enabled', False):
                hyperparams = hyperparams_config.get('param_grid', {})
                if hyperparams:
                    logger.info(f"Performing hyperparameter tuning for {model_name}")
                    # Perform tuning
                    model, best_params = model.hyperparameter_tuning(
                        X_train, y_train, param_grid=hyperparams, cv=self.n_folds
                    )
            else:
                logger.info(f"Skipping hyperparameter tuning for {model_name}")

            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, f"temp_{model_name}_{timestamp}.pkl")
            model.save(model_path)
            
            # Record results
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'model_path': model_path
            }
            
            # Update best model if needed
            if metrics['rmse'] < best_metric:
                best_metric = metrics['rmse']
                best_model = model_name
        
        # Train advanced models
        logger.info("Training Adaptive Temperature Regressor")
        adaptive_model = AdaptiveTemperatureRegressor()
        adaptive_model.fit(X_train, y_train)
        adaptive_metrics = self.model_evaluator.evaluate_regression_model(adaptive_model, X_test, y_test)
        
        # Save adaptive model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adaptive_path = os.path.join(self.models_dir, f"temp_adaptive_{timestamp}.pkl")
        joblib.dump(adaptive_model, adaptive_path)
        
        results['adaptive_temp'] = {
            'model': adaptive_model,
            'metrics': adaptive_metrics,
            'model_path': adaptive_path
        }
        
        logger.info("Training Deep Temperature Regressor")
        deep_model = DeepTemperatureRegressor()
        deep_model.fit(X_train, y_train)
        deep_metrics = self.model_evaluator.evaluate_regression_model(deep_model, X_test, y_test)
        
        # Save deep model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deep_path = os.path.join(self.models_dir, f"temp_deep_{timestamp}.pkl")
        joblib.dump(deep_model, deep_path)
        
        results['deep_temp'] = {
            'model': deep_model,
            'metrics': deep_metrics,
            'model_path': deep_path
        }
        
        # Update best model if any advanced model is better
        if adaptive_metrics['rmse'] < best_metric:
            best_metric = adaptive_metrics['rmse']
            best_model = 'adaptive_temp'
            
        if deep_metrics['rmse'] < best_metric:
            best_metric = deep_metrics['rmse']
            best_model = 'deep_temp'
        
        # Log best model
        logger.info(f"Best temperature prediction model: {best_model} with RMSE: {best_metric}")
        
        return results, best_model
    
    def train_plant_type_stage_models(self, features_data, target_column='Plant_Type_Stage'):
        """
        Train classification models to predict plant type-stage.
        
        Args:
            features_data (DataFrame): DataFrame with features
            target_column (str): Name of the plant type-stage column
            
        Returns:
            Dictionary containing trained models and their performance metrics
        """
        logger.info("Starting plant type-stage classification model training")
        
        # Prepare data
        data_splits = self.prepare_data(features_data, target_column)
        if len(data_splits) == 6:
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        else:
            X_train, X_test, y_train, y_test = data_splits
            X_val, y_val = None, None
        
        # Get enabled models from model_config
        enabled_models = [
            model_name for model_name, model_config in self.model_config['models'].items()
            if model_config.get('enabled', False)
        ]
        
        results = {}
        best_metric = 0  # Higher is better for classification metrics like accuracy
        best_model = None
        
        # Train standard models
        for model_name in enabled_models:
            logger.info(f"Training {model_name} for plant type-stage classification")
            
            # Create model
            model_params = self.model_config['models'][model_name].get('params', {})
            model = PlantTypeStageClassifier(model_name=model_name, model_params=model_params)
            
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Perform hyperparameter tuning if enabled
            hyperparams_config = self.model_config['models'][model_name].get('hyperparameter_tuning', {})
            if hyperparams_config.get('enabled', False):
                hyperparams = hyperparams_config.get('param_grid', {})
                if hyperparams:
                    logger.info(f"Performing hyperparameter tuning for {model_name}")
                    # Perform tuning
                    model, best_params = model.hyperparameter_tuning(
                        X_train, y_train, param_grid=hyperparams, cv=self.n_folds
                    )
            else:
                logger.info(f"Skipping hyperparameter tuning for {model_name}")

            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, f"plant_{model_name}_{timestamp}.pkl")
            model.save(model_path)
            
            # Record results
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'model_path': model_path
            }
            
            # Update best model if needed
            if metrics['accuracy'] > best_metric:
                best_metric = metrics['accuracy']
                best_model = model_name
        
        # Train advanced models
        logger.info("Training Ensemble Plant Classifier")
        ensemble_model = EnsemblePlantClassifier()
        ensemble_model.fit(X_train, y_train)
        ensemble_metrics = self.model_evaluator.evaluate_classification_model(ensemble_model, X_test, y_test)
        
        # Save ensemble model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_path = os.path.join(self.models_dir, f"plant_ensemble_{timestamp}.pkl")
        joblib.dump({'model': ensemble_model, 'label_encoder': ensemble_model.label_encoder}, ensemble_path)
        
        results['ensemble_plant'] = {
            'model': ensemble_model,
            'metrics': ensemble_metrics,
            'model_path': ensemble_path
        }
        
        logger.info("Training Adaptive Plant Classifier")
        # Find temperature feature index or name
        temp_feature = 'Temperature_Sensor_.C'
        temp_idx = None
        if temp_feature in X_train.columns:
            temp_idx = list(X_train.columns).index(temp_feature)
        
        adaptive_model = AdaptivePlantClassifier()
        adaptive_model.fit(X_train, y_train, temp_feature_name=temp_feature, temp_feature_idx=temp_idx)
        adaptive_metrics = self.model_evaluator.evaluate_classification_model(adaptive_model, X_test, y_test)
        
        # Save adaptive model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adaptive_path = os.path.join(self.models_dir, f"plant_adaptive_{timestamp}.pkl")
        joblib.dump({'model': adaptive_model, 'label_encoder': adaptive_model.label_encoder}, adaptive_path)
        
        results['adaptive_plant'] = {
            'model': adaptive_model,
            'metrics': adaptive_metrics,
            'model_path': adaptive_path
        }
        
        # Update best model if any advanced model is better
        if ensemble_metrics['f1_weighted'] > best_metric:
            best_metric = ensemble_metrics['f1_weighted']
            best_model = 'ensemble_plant'
            
        if adaptive_metrics['f1_weighted'] > best_metric:
            best_metric = adaptive_metrics['f1_weighted']
            best_model = 'adaptive_plant'
        
        # Log best model
        logger.info(f"Best plant type-stage model: {best_model} with F1: {best_metric}")
        
        return results, best_model
