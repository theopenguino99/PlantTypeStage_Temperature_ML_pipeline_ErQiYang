#!/usr/bin/env python3
"""
Main module to orchestrate the entire ML pipeline.
"""

import argparse
import os
import sys
from loguru import logger
from config_loader import load_config
from main_process import DataInput
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
        if not isinstance(config, dict):
            raise TypeError("load_config did not return a dictionary. Check its implementation.")

        # Create required directories
        for path in [config['paths']['data_dir'], config['paths']['models_dir'], config['paths']['results_dir']]:
            os.makedirs(path, exist_ok=True)
        
        # Load preprocessed data from main_process.py
        logger.info("Loading preprocessed data")
        input_regression_data = DataInput('regression')
        input_classification_data = DataInput('classification')

        # Train models
        logger.info("Training models")
        model_trainer = ModelTrainer()
        
        # Train temperature models (REGRESSION PROBLEM)
        temp_target_column = config['data']['target_num']
        temp_results, best_temp_model = model_trainer.train_temperature_models(input_regression_data)
        logger.info(f"Best temperature model: {best_temp_model}")
        
        # Train plant type-stage models (CLASSIFICATION PROBLEM)
        plant_target_column = config['data']['target_cat']
        plant_results, best_plant_model = model_trainer.train_plant_type_stage_models(input_classification_data, target_column=plant_target_column)
        logger.info(f"Best plant type-stage model: {best_plant_model}")
        
        # Save results
        results_dir = config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        temp_results_path = os.path.join(results_dir, "temperature_results.pkl")
        plant_results_path = os.path.join(results_dir, "plant_results.pkl")
        
        pd.to_pickle(temp_results, temp_results_path)
        pd.to_pickle(plant_results, plant_results_path)
        logger.info(f"Results saved to {results_dir}")
        
        logger.info("ML pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())


