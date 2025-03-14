from loguru import logger
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime
from config_loader import *


class ModelEvaluator:
    """
    Evaluates trained models and generates performance reports.
    """
    
    def __init__(self):
        """
        Initialize the model evaluator with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.results_dir = load_config()['paths']['results_dir']
        self.regression_metrics_regression = load_model_config()['evaluation_regression']['metrics']
        self.regression_metrics_classification = load_model_config()['evaluation_classification']['metrics']
        self.primary_metric_regression = load_model_config()['evaluation_regression']['primary_metric']
        self.primary_metric_classification = load_model_config()['evaluation_classification']['primary_metric']
        self.lower_is_better_regression = load_model_config()['evaluation_regression']['lower_is_better']
        self.lower_is_better_classification = load_model_config()['evaluation_classification']['lower_is_better']
        
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_regression_model(self, model, X_test, y_test):
        """
        Evaluate a regression model and compute various performance metrics.
        
        Args:
            model: Trained regression model
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of performance metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if 'r2' in self.regression_metrics_regression:
            metrics['r2'] = r2_score(y_test, y_pred)
        if 'mae' in self.regression_metrics_regression:
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
        if 'mse' in self.regression_metrics_regression:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
        if 'rmse' in self.regression_metrics_regression:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
        
        return metrics
    
    def evaluate_classification_model(self, model, X_test, y_test):
        """
        Evaluate a classification model and compute various performance metrics.
        
        Args:
            model: Trained classification model
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of performance metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if 'accuracy' in self.regression_metrics_regression:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        if 'precision' in self.regression_metrics_regression:
            metrics['precision'] = precision_score(y_test, y_pred, average=None)
        if 'recall' in self.regression_metrics_regression:
            metrics['recall'] = recall_score(y_test, y_pred, average=None)
        if 'f1' in self.regression_metrics_regression:
            metrics['f1_weighted'] = f1_score(y_test, y_pred, average=None)
        
        # Generate classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
        
        return metrics
    
    def save_evaluation_results(self, model_name, task_type, metrics, predictions=None):
        """
        Save evaluation results to disk.
        
        Args:
            model_name (str): Name of the model
            task_type (str): Type of task ('regression' or 'classification')
            metrics (dict): Performance metrics
            predictions (DataFrame, optional): DataFrame with predictions
            
        Returns:
            Path to the saved results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"{task_type}_{model_name}_{timestamp}")
        os.makedirs(results_path, exist_ok=True)
        
        # Save metrics
        metrics_copy = metrics.copy()
        # Convert complex objects to strings
        if 'classification_report' in metrics_copy and not isinstance(metrics_copy['classification_report'], dict):
            metrics_copy['classification_report'] = str(metrics_copy['classification_report'])
        
        with open(os.path.join(results_path, 'metrics.json'), 'w') as f:
            json.dump(metrics_copy, f, indent=4)
        
        # Save predictions if provided
        if predictions is not None:
            predictions.to_csv(os.path.join(results_path, 'predictions.csv'), index=False)
        
        logger.info(f"Evaluation results saved to {results_path}")
        return results_path