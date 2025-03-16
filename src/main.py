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
from pandas import DataFrame

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
        logger.opt(colors=True).info(f"<yellow>Loading configuration from {args.config}</yellow>")
        config = load_config()
        if not isinstance(config, dict):
            raise TypeError("load_config did not return a dictionary. Check its implementation.")
        
        # Load preprocessed data from main_process.py
        logger.opt(colors=True).info("<yellow>Load->Clean->Select->Engineer->Preprocess</yellow>")
        input_regression_data = DataInput('regression').process_data()
        input_classification_data = DataInput('classification').process_data()

        # Train models
        logger.opt(colors=True).info("<yellow>Training models</yellow>")
        model_trainer = ModelTrainer()
        
        # Train temperature models (REGRESSION PROBLEM)
        temp_results, best_temp_model = model_trainer.train_temperature_models(input_regression_data)
        logger.log("INFO", f"Best temperature model: {best_temp_model}", color="<yellow>")
        
        # Train plant type-stage models (CLASSIFICATION PROBLEM)
        plant_results, best_plant_model = model_trainer.train_plant_type_stage_models(input_classification_data)
        logger.info(f"Best plant type-stage model: {best_plant_model}")
        
        # Save results
        logger.opt(colors=True).info("<yellow>Saving results of Regression and Clasification problem</yellow>")
        results_dir = config['paths']['results_dir']
        
        # Convert results to DataFrames to save to a readable CSV format
        temp_results_df = DataFrame(temp_results)
        plant_results_df = DataFrame(plant_results)
        # Save results in a readable format (e.g., CSV)
        temp_results_df.to_csv(os.path.join(results_dir, "temperature_regression_results.csv"), index=False)
        plant_results_df.to_csv(os.path.join(results_dir, "plant_results.csv"), index=False)
        # Optionally, save as pickle for further processing
        pd.to_pickle(temp_results, os.path.join(results_dir, "temperature_regression_results.pkl"))
        pd.to_pickle(plant_results, os.path.join(results_dir, "plant_results.pkl"))
        
        logger.info(f"Results saved to {results_dir} in both CSV and pickle formats")
        
        logger.opt(colors = True).info("<green>ML Training pipeline completed successfully</green>")
        return 0
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())


