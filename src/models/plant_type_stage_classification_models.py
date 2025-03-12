import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import joblib
import os
import logging
from collections import Counter

class PlantTypeStageClassifier:
    """Base class for plant type-stage classification models in the agricultural pipeline."""
    
    def __init__(self, model_name, model_params=None, scaler=None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.scaler = scaler
        self.model = None
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
    
    def build_model(self):
        """Build the model based on model_name."""
        if self.model_name == "random_forest":
            model = RandomForestClassifier(**self.model_params)
        elif self.model_name == "gradient_boosting":
            model = GradientBoostingClassifier(**self.model_params)
        elif self.model_name == "xgboost":
            model = XGBClassifier(**self.model_params)
        elif self.model_name == "lightgbm":
            model = LGBMClassifier(**self.model_params)
        elif self.model_name == "mlp":
            model = MLPClassifier(**self.model_params)
        elif self.model_name == "svm":
            model = SVC(**self.model_params, probability=True)
        elif self.model_name == "logistic_regression":
            model = LogisticRegression(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
        
        # Create pipeline with scaling if needed
        if self.scaler is not None:
            self.model = Pipeline([
                ('scaler', self.scaler),
                ('classifier', model)
            ])
        else:
            self.model = model
        
        return self.model
    
    def train(self, X_train, y_train):
        """Train the model on the provided data."""
        if self.model is None:
            self.build_model()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        self.logger.info(f"Training {self.model_name} model for plant type-stage classification")
        self.logger.info(f"Class distribution: {Counter(y_train)}")
        self.model.fit(X_train, y_encoded)
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """Make probability predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, metrics=None):
        """Evaluate the model on test data."""
        metrics = metrics or ['accuracy', 'f1', 'precision', 'recall']
        y_encoded = self.label_encoder.transform(y_test)
        y_pred = self.predict(X_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        results = {}
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_encoded, y_pred_encoded)
        if 'f1' in metrics:
            results['f1_macro'] = f1_score(y_encoded, y_pred_encoded, average='macro')
            results['f1_weighted'] = f1_score(y_encoded, y_pred_encoded, average='weighted')
        if 'precision' in metrics:
            results['precision_macro'] = precision_score(y_encoded, y_pred_encoded, average='macro')
        if 'recall' in metrics:
            results['recall_macro'] = recall_score(y_encoded, y_pred_encoded, average='macro')
        
        # Add detailed classification report
        results['classification_report'] = classification_report(y_test, y_pred)
        
        # Add confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_encoded, y_pred_encoded)
        
        return results
    
    def save(self, filepath):
        """Save the model and label encoder to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model and label encoder from disk."""
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data['model_name']
        self.model_params = model_data['model_params']
        
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=5, n_jobs=-1, method='grid'):
        """Perform hyperparameter tuning for the model."""
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        if param_grid is None:
            # Define default param grid based on model type
            if self.model_name == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_name in ["xgboost", "lightgbm"]:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            else:
                self.logger.warning(f"No default param grid for {self.model_name}. Using empty grid.")
                param_grid = {}
        
        # Create a base model
        base_model = self.build_model()
        
        # Set up the search
        if method == 'grid':
            search = GridSearchCV(base_model, param_grid, cv=cv, n_jobs=n_jobs,
                                 scoring='f1_macro')
        else:  # random search
            search = RandomizedSearchCV(base_model, param_grid, cv=cv, n_jobs=n_jobs,
                                       scoring='f1_macro', n_iter=20)
        
        self.logger.info(f"Performing {method} search for hyperparameter tuning")
        search.fit(X, y_encoded)
        
        # Update model with best parameters
        self.logger.info(f"Best parameters: {search.best_params_}")
        
        # Update model parameters and rebuild model
        if hasattr(search, 'best_estimator_'):
            self.model = search.best_estimator_
        else:
            # Extract best parameters and update model_params
            if self.scaler is not None:
                # For pipeline, need to extract classifier params
                best_params = {k.replace('classifier__', ''): v 
                             for k, v in search.best_params_.items() 
                             if k.startswith('classifier__')}
            else:
                best_params = search.best_params_
                
            self.model_params.update(best_params)
            self.build_model()
            self.model.fit(X, y_encoded)
        
        return self, search.best_params_


class EnsemblePlantClassifier(BaseEstimator, ClassifierMixin):
    """
    An ensemble classifier that combines multiple models for plant type-stage classification
    using voting or stacking techniques.
    """
    
    def __init__(self, base_models=None, voting='soft', weights=None):
        self.base_models = base_models or [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            XGBClassifier(n_estimators=100, random_state=42)
        ]
        self.voting = voting  # 'hard' or 'soft'
        self.weights = weights
        self.label_encoder = LabelEncoder()
        self.trained_models = None
        self.logger = logging.getLogger(__name__)
        self.classes_ = None
    
    def fit(self, X, y):
        """Train all base models on the same data."""
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Train all base models
        self.trained_models = []
        for i, model in enumerate(self.base_models):
            self.logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            model.fit(X, y_encoded)
            self.trained_models.append(model)
        
        return self
    
    def predict(self, X):
        """Make predictions using voting from all base models."""
        if self.trained_models is None:
            raise ValueError("Models have not been trained yet.")
        
        if self.voting == 'hard':
            # Get predictions from each model
            predictions = np.array([model.predict(X) for model in self.trained_models])
            
            # Transpose to get predictions per sample
            predictions = predictions.T
            
            # Use majority voting
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x, weights=self.weights).argmax(), 
                axis=1, 
                arr=predictions
            )
        else:  # soft voting
            # Get probability predictions
            probas = self.predict_proba(X)
            final_pred = np.argmax(probas, axis=1)
        
        # Convert back to original labels
        return self.label_encoder.inverse_transform(final_pred)
    
    def predict_proba(self, X):
        """Make probability predictions using averaging from all base models."""
        if self.trained_models is None:
            raise ValueError("Models have not been trained yet.")
        
        # Get probability predictions from each model
        all_probas = []
        for i, model in enumerate(self.trained_models):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba)
        
        # Average probabilities (with weights if provided)
        if self.weights:
            weights = np.array(self.weights)
            weights = weights / weights.sum()  # Normalize weights
            final_probas = np.average(np.array(all_probas), axis=0, weights=weights)
        else:
            final_probas = np.mean(np.array(all_probas), axis=0)
        
        return final_probas


class AdaptivePlantClassifier(BaseEstimator, ClassifierMixin):
    """
    An adaptive classifier that adjusts its behavior based on sensor readings
    and time-based features for plant type-stage classification.
    """
    
    def __init__(self, models_config=None):
        """
        Initialize the adaptive classifier with specialized models for different conditions.
        
        Args:
            models_config: Dictionary mapping condition ranges to specific models
        """
        if models_config is None:
            # Default configuration with temperature ranges and corresponding models
            self.models_config = {
                'low_temp': {
                    'range': (float('-inf'), 15.0),  # Temperature < 15°C
                    'model': RandomForestClassifier(n_estimators=150, random_state=42)
                },
                'medium_temp': {
                    'range': (15.0, 25.0),  # 15°C <= Temperature < 25°C
                    'model': GradientBoostingClassifier(n_estimators=100, random_state=42)
                },
                'high_temp': {
                    'range': (25.0, float('inf')),  # Temperature >= 25°C
                    'model': XGBClassifier(n_estimators=100, random_state=42)
                }
            }
        else:
            self.models_config = models_config
            
        self.trained_models = {}
        self.default_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.temp_feature_idx = None  # Will be determined during fit
        self.logger = logging.getLogger(__name__)
        self.classes_ = None
        
    def _get_model_for_conditions(self, X_sample):
        """Determine which model to use based on environmental conditions."""
        if self.temp_feature_idx is None:
            return self.default_model
            
        temperature = X_sample[self.temp_feature_idx]
        
        for config_name, config in self.models_config.items():
            low, high = config['range']
            if low <= temperature < high:
                return self.trained_models.get(config_name, self.default_model)
                
        return self.default_model
    
    def fit(self, X, y, temp_feature_name=None, temp_feature_idx=None):
        """
        Train specialized models for different environmental conditions.
        
        Args:
            X: Feature matrix
            y: Target labels
            temp_feature_name: Name of the temperature feature (if X is DataFrame)
            temp_feature_idx: Index of the temperature feature (if X is numpy array)
        """
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Determine temperature feature index
        if temp_feature_name is not None and hasattr(X, 'columns'):
            self.temp_feature_idx = list(X.columns).index(temp_feature_name)
        elif temp_feature_idx is not None:
            self.temp_feature_idx = temp_feature_idx
        else:
            self.logger.warning("Temperature feature not specified. Using default model for all predictions.")
            self.default_model.fit(X, y_encoded)
            return self
        
        # Train specialized models for each condition range
        for config_name, config in self.models_config.items():
            low, high = config['range']
            temp_values = X[:, self.temp_feature_idx] if isinstance(X, np.ndarray) else X.iloc[:, self.temp_feature_idx]
            
            # Filter data for this temperature range
            mask = (temp_values >= low) & (temp_values < high)
            if np.sum(mask) > 10:  # Only train if we have enough samples
                X_subset = X[mask] if isinstance(X, np.ndarray) else X.iloc[mask]
                y_subset = y_encoded[mask]
                
                self.logger.info(f"Training model for {config_name} with {np.sum(mask)} samples")
                model = config['model']
                model.fit(X_subset, y_subset)
                self.trained_models[config_name] = model
            else:
                self.logger.warning(f"Not enough samples for {config_name} ({np.sum(mask)} samples)")
        
        # Train default model on all data
        self.default_model.fit(X, y_encoded)
        
        return self
    
    def predict(self, X):
        """Make predictions using the appropriate model for each sample's conditions."""
        predictions = np.zeros(X.shape[0], dtype=int)
        
        # For each sample, select the appropriate model and predict
        for i in range(X.shape[0]):
            X_sample = X[i] if isinstance(X, np.ndarray) else X.iloc[i]
            model = self._get_model_for_conditions(X_sample)
            
            # Reshape for single sample prediction
            X_sample_reshaped = X_sample.reshape(1, -1) if isinstance(X_sample, np.ndarray) else pd.DataFrame([X_sample])
            predictions[i] = model.predict(X_sample_reshaped)[0]
            
        # Convert back to original labels
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """Make probability predictions using the appropriate model for each sample."""
        n_classes = len(self.classes_)
        probas = np.zeros((X.shape[0], n_classes))
        
        # For each sample, select the appropriate model and predict probabilities
        for i in range(X.shape[0]):
            X_sample = X[i] if isinstance(X, np.ndarray) else X.iloc[i]
            model = self._get_model_for_conditions(X_sample)
            
            # Reshape for single sample prediction
            X_sample_reshaped = X_sample.reshape(1, -1) if isinstance(X_sample, np.ndarray) else pd.DataFrame([X_sample])
            
            if hasattr(model, 'predict_proba'):
                probas[i] = model.predict_proba(X_sample_reshaped)[0]
            else:
                # For models without predict_proba, use one-hot encoding based on prediction
                pred = model.predict(X_sample_reshaped)[0]
                probas[i, pred] = 1.0
                
        return probas
