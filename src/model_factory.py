import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from .temperature_regression_models import (
    TemperatureRegressionModel, 
    AdaptiveTemperatureRegressor,
    DeepTemperatureRegressor
)
from .plant_type_stage_classification_models import (
    PlantTypeStageClassifier,
    EnsemblePlantClassifier,
    AdaptivePlantClassifier
)

class ModelFactory:
    """
    Factory class to create and configure models for the ML pipeline.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_model(self, model_type, task_type, use_scaler=True):
        """
        Create a model instance based on model type and task type.
        
        Args:
            model_type (str): Type of model to create (e.g., 'random_forest', 'xgboost', etc.)
            task_type (str): Type of task ('regression' for temperature, 'classification' for plant type-stage)
            use_scaler (bool): Whether to use a scaler with the model
            
        Returns:
            Model instance
        """
        scaler = StandardScaler() if use_scaler else None
        
        # Get common parameters
        common_params = self.config.get('common', {})
        
        # Get model-specific parameters
        model_config = self.config.get('models', {}).get(model_type, {})
        model_params = model_config.get('params', {})
        
        # Merge common and model-specific parameters
        params = {**common_params, **model_params}
        
        if task_type == 'regression':
            return TemperatureRegressionModel(model_name=model_type, model_params=params, scaler=scaler)
        elif task_type == 'classification':
            return PlantTypeStageClassifier(model_name=model_type, model_params=params, scaler=scaler)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def create_advanced_model(self, model_type, task_type, **kwargs):
        """
        Create an advanced model instance based on model type and task type.
        
        Args:
            model_type (str): Type of advanced model ('adaptive', 'ensemble', 'deep')
            task_type (str): Type of task ('regression' for temperature, 'classification' for plant type-stage)
            **kwargs: Additional parameters specific to the advanced model
            
        Returns:
            Advanced model instance
        """
        if task_type == 'regression':
            if model_type == 'adaptive':
                return AdaptiveTemperatureRegressor(**kwargs)
            elif model_type == 'deep':
                return DeepTemperatureRegressor(**kwargs)
            else:
                raise ValueError(f"Unknown advanced regression model type: {model_type}")
        elif task_type == 'classification':
            if model_type == 'ensemble':
                return EnsemblePlantClassifier(**kwargs)
            elif model_type == 'adaptive':
                return AdaptivePlantClassifier(**kwargs)
            else:
                raise ValueError(f"Unknown advanced classification model type: {model_type}")
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_enabled_models(self, task_type):
        """
        Get a list of models enabled in the configuration for a specific task type.
        
        Args:
            task_type (str): Type of task ('regression' or 'classification')
            
        Returns:
            List of enabled model names
        """
        enabled_models = []
        
        for model_name, model_config in self.config.get('models', {}).items():
            if model_config.get('enabled', False):
                enabled_models.append(model_name)
        
        self.logger.info(f"Enabled models for {task_type}: {enabled_models}")
        return enabled_models
    
    def get_model_hyperparameters(self, model_name):
        """
        Get hyperparameter tuning configuration for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dictionary with hyperparameter tuning configuration or None
        """
        model_config = self.config.get('models', {}).get(model_name, {})
        tuning_config = model_config.get('hyperparameter_tuning', {})
        
        if not tuning_config.get('enabled', False):
            return None
            
        return tuning_config.get('param_grid', {})
