import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from loguru import logger
from config_loader import load_model_config
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from config_loader import load_config, load_model_config


class BaseTemperatureRegressor(BaseEstimator, RegressorMixin):
    """
    Base class for temperature regression models, providing shared functionality.
    """
    def __init__(self):
        self.model = None
        self.config = load_config()  # Load configuration
        self.model_config = load_model_config()  # Load model configuration

    def evaluate(self, X_test, y_test, metrics=None):
        """Evaluate the model on test data."""
        # Use metrics from the config file if not explicitly provided
        metrics = metrics or self.model_config['evaluation_regression']['metrics']
        predictions = self.model.predict(X_test)
        
        results = {}
        if 'r2' in metrics:
            results['r2'] = r2_score(y_test, predictions)
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_test, predictions)
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(y_test, predictions)
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Log metrics
        for metric_name, metric_value in results.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
        
        return results

    def save(self, filepath):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load the model from disk."""
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model


class TemperatureRegressionModel(BaseTemperatureRegressor):
    """Standard temperature regression model."""
    
    def __init__(self, model_name, model_params=None):
        super().__init__()
        self.model_name = model_name
        self.model_params = model_params or {}

    def build_model(self):
        """Build the model based on model_name."""
        if self.model_name == "random_forest":
            model = RandomForestRegressor(**self.model_params)
        elif self.model_name == "gradient_boosting":
            model = GradientBoostingRegressor(**self.model_params)
        elif self.model_name == "xgboost":
            model = XGBRegressor(**self.model_params)
        elif self.model_name == "linear_regression":
            model = LinearRegression(**self.model_params)
        elif self.model_name == "ridge_regression":
            model = Ridge(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
        
        self.model = model
        return self.model

    def train(self, X_train, y_train):
        """Train the model on the provided data."""
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        return self.model

class DeepTemperatureRegressor(BaseTemperatureRegressor):
    """
    A deep neural network for temperature prediction with configurable architecture.
    """
    
    def __init__(self):
        super().__init__()
        # Load parameters from the configuration file
        deep_model_config = self.model_config['advanced_models']['deep_temperature']
        self.hidden_layers = tuple(deep_model_config['params']['hidden_layers'])
        self.activation = deep_model_config['params']['activation']
        self.solver = deep_model_config['params']['solver']
        self.learning_rate = deep_model_config['params']['learning_rate']
        self.max_iter = deep_model_config['params']['max_iter']
        self.alpha = deep_model_config['params']['alpha']
        self.random_state = self.config['common']['random_state']
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Train the deep neural network model."""
        # Scale the input data
        X_scaled = self.scaler.fit_transform(X)
        
        # Configure the model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver=self.solver,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained deep neural network."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=5, n_jobs=-1, method='grid'):
        """Perform hyperparameter tuning for the model."""
        X_scaled = self.scaler.fit_transform(X)
        
        if param_grid is None:
            param_grid = self.model_config['advanced_models']['deep_temperature']['hyperparameter_tuning']['param_grid']
        
        model = MLPRegressor(random_state=self.random_state)
        
        if method == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, 
                                scoring='neg_mean_squared_error')
        else:  # random search
            search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, 
                                       scoring='neg_mean_squared_error', n_iter=20)
        
        logger.info(f"Performing {method} search for hyperparameter tuning")
        search.fit(X_scaled, y)
        
        # Update model with best parameters
        logger.info(f"Best parameters: {search.best_params_}")
        self.hidden_layers = search.best_params_.get('hidden_layer_sizes', self.hidden_layers)
        self.activation = search.best_params_.get('activation', self.activation)
        self.alpha = search.best_params_.get('alpha', self.alpha)
        self.learning_rate = search.best_params_.get('learning_rate', self.learning_rate)
        self.max_iter = search.best_params_.get('max_iter', self.max_iter)
        
        # Create the model with best parameters
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver=self.solver,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=self.random_state
        )
        
        # Train the model with the best parameters
        self.model.fit(X_scaled, y)
        
        return self.model, search.best_params_