#!/usr/bin/env python3
"""
Main module to orchestrate the entire ML pipeline.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add the src directory to sys.path

from loguru import logger
from config_loader import load_config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from visualization.visualizer import Visualizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the ML pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to the configuration file')
    return parser.parse_args()

def main():
    """Main function to run the ML pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Create required directories
        for path in [config['paths']['data_dir'], config['paths']['models_dir'], 
                    config['paths']['results_dir']]:
            os.makedirs(path, exist_ok=True)
        
        # Load data
        logger.info("Loading data")
        data_loader = DataLoader(config)
        data = data_loader.load_data()
        
        # Clean data
        logger.info("Cleaning data")
        data_cleaner = DataCleaner(config)
        cleaned_data = data_cleaner.clean_data(data)
        
        # Feature selection
        logger.info("Selecting features")
        feature_selector = FeatureSelector(config)
        X_train, X_val, X_test, y_train, y_val, y_test = feature_selector.select_features(feature_data)
        
        # Feature engineering
        logger.info("Engineering features")
        feature_engineer = FeatureEngineer(config)
        feature_data = feature_engineer.engineer_features(preprocessed_data)
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor(config)
        preprocessed_data = preprocessor.preprocess(cleaned_data)
        
        # Train models
        logger.info("Training models")
        model_trainer = ModelTrainer(config)
        trained_models = model_trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        logger.info("Evaluating models")
        model_evaluator = ModelEvaluator(config)
        evaluation_results, best_model = model_evaluator.evaluate_models(
            trained_models, X_test, y_test
        )
        
        # Generate visualizations
        logger.info("Generating visualizations")
        visualizer = Visualizer(config)
        visualizer.create_visualizations(data, cleaned_data, feature_data, evaluation_results, best_model)
        
        # Save best model
        logger.info("Saving best model")
        model_path = os.path.join(config['paths']['models_dir'], 'best_model.joblib')
        model_trainer.save_model(best_model, model_path)
        
        logger.info("ML pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Test the main function


