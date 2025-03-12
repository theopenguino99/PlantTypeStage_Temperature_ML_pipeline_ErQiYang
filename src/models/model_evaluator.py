import numpy as np
import pandas as pd
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime

class ModelEvaluator:
    """
    Evaluates trained models and generates performance reports.
    """
    
    def __init__(self, config):
        """
        Initialize the model evaluator with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.results_dir = config.get('paths', {}).get('results_dir', 'results/')
        self.logger = logging.getLogger(__name__)
        self.regression_metrics = config.get('evaluation', {}).get('metrics', ['r2', 'mae', 'mse', 'rmse'])
        self.primary_metric = config.get('evaluation', {}).get('primary_metric', 'rmse')
        self.lower_is_better = config.get('evaluation', {}).get('lower_is_better', True)
        
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
        if 'r2' in self.regression_metrics:
            metrics['r2'] = r2_score(y_test, y_pred)
        if 'mae' in self.regression_metrics:
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
        if 'mse' in self.regression_metrics:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
        if 'rmse' in self.regression_metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Additional metrics can be added here
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
        
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
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log metrics
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        
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
        
        self.logger.info(f"Evaluation results saved to {results_path}")
        return results_path
    
    def plot_regression_results(self, y_true, y_pred, model_name, save_path=None):
        """
        Generate and save plots for regression results.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name (str): Name of the model
            save_path (str, optional): Path to save the plots
            
        Returns:
            List of paths to saved plots
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f"regression_{model_name}_{timestamp}")
            os.makedirs(save_path, exist_ok=True)
        
        saved_plots = []
        
        # 1. Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name}: Actual vs Predicted')
        
        actual_vs_pred_path = os.path.join(save_path, 'actual_vs_predicted.png')
        plt.savefig(actual_vs_pred_path)
        plt.close()
        saved_plots.append(actual_vs_pred_path)
        
        # 2. Residual Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.title(f'{model_name}: Residual Plot')
        
        residual_path = os.path.join(save_path, 'residual_plot.png')
        plt.savefig(residual_path)
        plt.close()
        saved_plots.append(residual_path)
        
        # 3. Residual Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Residual Distribution')
        
        residual_dist_path = os.path.join(save_path, 'residual_distribution.png')
        plt.savefig(residual_dist_path)
        plt.close()
        saved_plots.append(residual_dist_path)
        
        return saved_plots
    
    def plot_classification_results(self, y_true, y_pred, model_name, class_names=None, save_path=None):
        """
        Generate and save plots for classification results.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name (str): Name of the model
            class_names (list, optional): List of class names
            save_path (str, optional): Path to save the plots
            
        Returns:
            List of paths to saved plots
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f"classification_{model_name}_{timestamp}")
            os.makedirs(save_path, exist_ok=True)
        
        saved_plots = []
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name}: Confusion Matrix')
        
        cm_path = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        saved_plots.append(cm_path)
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=y_true)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(f'Class Distribution (Actual)')
        if class_names:
            plt.xticks(range(len(class_names)), class_names, rotation=45)
        
        class_dist_path = os.path.join(save_path, 'class_distribution.png')
        plt.savefig(class_dist_path)
        plt.close()
        saved_plots.append(class_dist_path)
        
        return saved_plots
    
    def compare_models(self, models_results, task_type, primary_metric=None):
        """
        Compare multiple models based on their performance metrics.
        
        Args:
            models_results (dict): Dictionary of model results
            task_type (str): Type of task ('regression' or 'classification')
            primary_metric (str, optional): Primary metric for comparison
            
        Returns:
            DataFrame with model comparison
        """
        if primary_metric is None:
            primary_metric = self.primary_metric if task_type == 'regression' else 'f1_weighted'
        
        comparison = []
        for model_name, result in models_results.items():
            metrics = result['metrics']
            
            if task_type == 'regression':
                row = {
                    'model': model_name,
                    'r2': metrics.get('r2', None),
                    'mae': metrics.get('mae', None),
                    'mse': metrics.get('mse', None),
                    'rmse': metrics.get('rmse', None)
                }
            else:  # classification
                row = {
                    'model': model_name,
                    'accuracy': metrics.get('accuracy', None),
                    'precision_macro': metrics.get('precision_macro', None),
                    'recall_macro': metrics.get('recall_macro', None),
                    'f1_macro': metrics.get('f1_macro', None),
                    'f1_weighted': metrics.get('f1_weighted', None)
                }
            
            comparison.append(row)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        # Sort by primary metric
        ascending = self