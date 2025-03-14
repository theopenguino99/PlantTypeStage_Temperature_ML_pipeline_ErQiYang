#!/usr/bin/env python3
"""
Main module to orchestrate the entire ML pipeline.
"""

import argparse
import os
import sys
from loguru import logger
from config_loader import load_config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from model_trainer import ModelTrainer
import pandas as pd

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
        config = load_config()
        
        # Create required directories
        for path in [config['paths']['data_dir'], config['paths']['models_dir'], 
                     config['paths']['results_dir']]:
            os.makedirs(path, exist_ok=True)
        
        # Load data
        logger.info("Loading data")
        data_loader = DataLoader()
        data = data_loader.load_data()
        
        # Clean data
        logger.info("Cleaning data")
        data_cleaner = DataCleaner()
        cleaned_data = data_cleaner.clean_data(data)
        
        # Feature selection
        logger.info("Selecting features")
        feature_selector = FeatureSelector()
        data_selected = feature_selector.select_features(cleaned_data)
        
        # Feature engineering
        logger.info("Engineering features")
        feature_engineer = FeatureEngineer()
        data_engineered = feature_engineer.engineer_features(data_selected)
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        preprocessed_data = preprocessor.preprocess(data_engineered)
        
        # Train models
        logger.info("Training models")
        model_trainer = ModelTrainer()
        
        # Train temperature models
        temp_target_column = config['data']['target_num']
        temp_results, best_temp_model = model_trainer.train_temperature_models(preprocessed_data) # Error while training here
        logger.info(f"Best temperature model: {best_temp_model}")
        
        # # Train plant type-stage models
        # plant_target_column = config['data']['target_cat']
        # plant_results, best_plant_model = model_trainer.train_plant_type_stage_models(preprocessed_data, target_column=plant_target_column)
        # logger.info(f"Best plant type-stage model: {best_plant_model}")
        
        # Save results
        results_dir = config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        temp_results_path = os.path.join(results_dir, "temperature_results.pkl")
        plant_results_path = os.path.join(results_dir, "plant_results.pkl")
        
        pd.to_pickle(temp_results, temp_results_path)
        # pd.to_pickle(plant_results, plant_results_path)
        logger.info(f"Results saved to {results_dir}")
        
        logger.info("ML pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())


