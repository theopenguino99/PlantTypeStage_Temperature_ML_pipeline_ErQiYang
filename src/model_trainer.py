import numpy as np
import pandas as pd
import logging
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime
from .model_factory import ModelFactory
from .model_evaluator import ModelEvaluator

class ModelTrainer:
    """
    Trains machine learning models for both temperature prediction and plant type-stage classification.
    """
    
    def __init__(self, config):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.pipeline_config = config.get('pipeline', {})
        self.data_config = config.get('data', {})
        self.models_dir = config.get('paths', {}).get('models_dir', 'models/')
        self.use_cross_validation = self.pipeline_config.get('use_cross_validation', False)
        self.n_folds = self.pipeline_config.get('n_folds', 5)
        self.model_factory = ModelFactory(config.get('model_config', {}))
        self.model_evaluator = ModelEvaluator(config)
        self.logger = logging.getLogger(__name__)
        
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
        test_size = self.data_config.get('test_size', 0.2)
        random_state = self.data_config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # If validation set is required
        validation_size = self.data_config.get('validation_size', 0.25)
        if validation_size > 0:
            train_size = 1 - validation_size
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=train_size, random_state=random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_temperature_models(self, features_data, target_column='Temperature_Sensor_.C'):
        """
        Train regression models to predict temperature.
        
        Args:
            features_data (DataFrame): DataFrame with features
            target_column (str): Name of the temperature column
            
        Returns:
            Dictionary containing trained models and their performance metrics
        """
        self.logger.info("Starting temperature prediction model training")
        
        # Prepare data
        data_splits = self.prepare_data(features_data, target_column)
        if len(data_splits) == 6:
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        else:
            X_train, X_test, y_train, y_test = data_splits
            X_val, y_val = None, None
        
        # Get enabled models from config
        enabled_models = self.model_factory.get_enabled_models('regression')
        
        results = {}
        best_metric = float('inf')  # Lower is better for regression metrics like RMSE
        best_model = None
        
        # Train standard models
        for model_name in enabled_models:
            self.logger.info(f"Training {model_name} for temperature prediction")
            
            # Create model
            model = self.model_factory.create_model(model_name, 'regression')
            
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Perform hyperparameter tuning if enabled
            hyperparams = self.model_factory.get_model_hyperparameters(model_name)
            if hyperparams:
                self.logger.info(f"Performing hyperparameter tuning for {model_name}")
                
                # Create a fresh model for tuning
                tuning_model = self.model_factory.create_model(model_name, 'regression')
                
                # Prepare tuning data (use validation set if available)
                X_tune = X_val if X_val is not None else X_train
                y_tune = y_val if y_val is not None else y_train
                
                # Perform tuning
                tuning_model, best_params = tuning_model.hyperparameter_tuning(
                    X_tune, y_tune, param_grid=hyperparams, cv=self.n_folds
                )
                
                # Evaluate tuned model
                tuning_metrics = tuning_model.evaluate(X_test, y_test)
                
                # If tuned model is better, use it
                if tuning_metrics['rmse'] < metrics['rmse']:
                    model = tuning_model
                    metrics = tuning_metrics
                    self.logger.info(f"Tuned model is better: {metrics}")
            
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
        # 1. Adaptive Temperature Regressor
        self.logger.info("Training Adaptive Temperature Regressor")
        adaptive_model = self.model_factory.create_advanced_model('adaptive', 'regression')
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
        
        # 2. Deep Temperature Regressor
        self.logger.info("Training Deep Temperature Regressor")
        deep_model = self.model_factory.create_advanced_model('deep', 'regression')
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
        self.logger.info(f"Best temperature prediction model: {best_model} with RMSE: {best_metric}")
        
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
        self.logger.info("Starting plant type-stage classification model training")
        
        # Prepare data
        data_splits = self.prepare_data(features_data, target_column)
        if len(data_splits) == 6:
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        else:
            X_train, X_test, y_train, y_test = data_splits
            X_val, y_val = None, None
        
        # Get enabled models from config
        enabled_models = self.model_factory.get_enabled_models('classification')
        
        results = {}
        best_metric = 0  # Higher is better for classification metrics like accuracy
        best_model = None
        
        # Train standard models
        for model_name in enabled_models:
            self.logger.info(f"Training {model_name} for plant type-stage classification")
            
            # Create model
            model = self.model_factory.create_model(model_name, 'classification')
            
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Perform hyperparameter tuning if enabled
            hyperparams = self.model_factory.get_model_hyperparameters(model_name)
            if hyperparams:
                self.logger.info(f"Performing hyperparameter tuning for {model_name}")
                
                # Create a fresh model for tuning
                tuning_model = self.model_factory.create_model(model_name, 'classification')
                
                # Prepare tuning data (use validation set if available)
                X_tune = X_val if X_val is not None else X_train
                y_tune = y_val if y_val is not None else y_train
                
                # Perform tuning
                tuning_model, best_params = tuning_model.hyperparameter_tuning(
                    X_tune, y_tune, param_grid=hyperparams, cv=self.n_folds
                )
                
                # Evaluate tuned model
                tuning_metrics = tuning_model.evaluate(X_test, y_test)
                
                # If tuned model is better, use it
                if tuning_metrics['f1_weighted'] > metrics['f1_weighted']:
                    model = tuning_model
                    metrics = tuning_metrics
                    self.logger.info(f"Tuned model is better: {metrics}")
            
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
            if metrics['f1_weighted'] > best_metric:
                best_metric = metrics['f1_weighted']
                best_model = model_name
        
        # Train advanced models
        # 1. Ensemble Plant Classifier
        self.logger.info("Training Ensemble Plant Classifier")
        ensemble_model = self.model_factory.create_advanced_model('ensemble', 'classification')
        ensemble_model.fit(X_train, y_train)
        ensemble_metrics = self.model_evaluator.evaluate_classification_model(ensemble_model, X_test, y_test)
        
        # Save ensemble model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_path = os.path.join(self.models_dir, f"plant_ensemble_{timestamp}.pkl")
        model_data = {
            'model': ensemble_model,
            'label_encoder': ensemble_model.label_encoder
        }
        joblib.dump(model_data, ensemble_path)
        
        results['ensemble_plant'] = {
            'model': ensemble_model,
            'metrics': ensemble_metrics,
            'model_path': ensemble_path
        }
        
        # 2. Adaptive Plant Classifier
        self.logger.info("Training Adaptive Plant Classifier")
        # Find temperature feature index or name
        temp_feature = 'Temperature_Sensor_.C'
        temp_idx = None
        if temp_feature in X_train.columns:
            temp_idx = list(X_train.columns).index(temp_feature)
        
        adaptive_model = self.model_factory.create_advanced_model('adaptive', 'classification')
        adaptive_model.fit(X_train, y_train, temp_feature_name=temp_feature, temp_feature_idx=temp_idx)
        adaptive_metrics = self.model_evaluator.evaluate_classification_model(adaptive_model, X_test, y_test)
        
        # Save adaptive model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adaptive_path = os.path.join(self.models_dir, f"plant_adaptive_{timestamp}.pkl")
        model_data = {
            'model': adaptive_model,
            'label_encoder': adaptive_model.label_encoder
        }
        joblib.dump(model_data, adaptive_path)
        
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
        self.logger.info(f"Best plant type-stage model: {best_model} with F1: {best_metric}")
        
        return results, best_model
