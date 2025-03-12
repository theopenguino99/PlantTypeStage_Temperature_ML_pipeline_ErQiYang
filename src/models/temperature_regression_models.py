import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import logging

class TemperatureRegressionModel:
    """Base class for temperature regression models in the agricultural pipeline."""
    
    def __init__(self, model_name, model_params=None, scaler=None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.scaler = scaler
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def build_model(self):
        """Build the model based on model_name."""
        if self.model_name == "random_forest":
            model = RandomForestRegressor(**self.model_params)
        elif self.model_name == "gradient_boosting":
            model = GradientBoostingRegressor(**self.model_params)
        elif self.model_name == "xgboost":
            model = XGBRegressor(**self.model_params)
        elif self.model_name == "lightgbm":
            model = LGBMRegressor(**self.model_params)
        elif self.model_name == "mlp":
            model = MLPRegressor(**self.model_params)
        elif self.model_name == "svr":
            model = SVR(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
        
        # Create pipeline with scaling if needed
        if self.scaler is not None:
            self.model = Pipeline([
                ('scaler', self.scaler),
                ('regressor', model)
            ])
        else:
            self.model = model
        
        return self.model
    
    def train(self, X_train, y_train):
        """Train the model on the provided data."""
        if self.model is None:
            self.build_model()
        
        self.logger.info(f"Training {self.model_name} model for temperature prediction")
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, metrics=None):
        """Evaluate the model on test data."""
        metrics = metrics or ['r2', 'mae', 'mse', 'rmse']
        predictions = self.predict(X_test)
        
        results = {}
        if 'r2' in metrics:
            results['r2'] = r2_score(y_test, predictions)
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_test, predictions)
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(y_test, predictions)
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        
        return results
    
    def save(self, filepath):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model from disk."""
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        return self


class AdaptiveTemperatureRegressor(BaseEstimator, RegressorMixin):
    """
    An adaptive model that combines multiple regressors for temperature prediction
    based on environmental conditions and time-based features.
    """
    
    def __init__(self, base_models=None, meta_model=None, use_features_importance=True):
        self.base_models = base_models or [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            XGBRegressor(n_estimators=100, random_state=42),
        ]
        self.meta_model = meta_model or LGBMRegressor(n_estimators=100, random_state=42)
        self.use_features_importance = use_features_importance
        self.trained_base_models = None
        self.feature_importances_ = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y):
        """Train the base models and the meta-model stacking them together."""
        self.trained_base_models = []
        
        # Train base models
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            self.logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            model.fit(X, y)
            base_predictions[:, i] = model.predict(X)
            self.trained_base_models.append(model)
        
        # Get feature importances if available
        if self.use_features_importance:
            self.calculate_feature_importances(X)
        
        # Combine base predictions with original features for meta-model
        meta_features = np.hstack((base_predictions, X))
        
        # Train meta-model
        self.logger.info("Training meta-model")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained stacked model."""
        if self.trained_base_models is None:
            raise ValueError("Model has not been trained yet.")
        
        # Generate base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.trained_base_models)))
        for i, model in enumerate(self.trained_base_models):
            base_predictions[:, i] = model.predict(X)
        
        # Combine with original features for final prediction
        meta_features = np.hstack((base_predictions, X))
        
        return self.meta_model.predict(meta_features)
    
    def calculate_feature_importances(self, X):
        """Calculate feature importances from base models if available."""
        feature_importances = np.zeros(X.shape[1])
        count = 0
        
        for model in self.trained_base_models:
            if hasattr(model, 'feature_importances_'):
                feature_importances += model.feature_importances_
                count += 1
        
        if count > 0:
            self.feature_importances_ = feature_importances / count
    
    def get_feature_importance(self, feature_names=None):
        """Return the feature importances in a readable format."""
        if self.feature_importances_ is None:
            return None
        
        if feature_names is None:
            return self.feature_importances_
        
        return dict(zip(feature_names, self.feature_importances_))


class DeepTemperatureRegressor(BaseEstimator, RegressorMixin):
    """
    A deep neural network for temperature prediction with configurable architecture.
    """
    
    def __init__(self, hidden_layers=(100, 50), activation='relu', solver='adam', 
                 learning_rate='adaptive', max_iter=1000, alpha=0.0001, random_state=42):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.logger = logging.getLogger(__name__)
    
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
        
        self.logger.info("Training deep neural network model")
        self.model.fit(X_scaled, y)
        return self
    
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
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000, 2000]
            }
        
        model = MLPRegressor(random_state=self.random_state)
        
        if method == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, 
                                scoring='neg_mean_squared_error')
        else:  # random search
            search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, 
                                       scoring='neg_mean_squared_error', n_iter=20)
        
        self.logger.info(f"Performing {method} search for hyperparameter tuning")
        search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.logger.info(f"Best parameters: {search.best_params_}")
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
        
        return self, search.best_params_
