import argparse
import joblib
import os
from data_loader import DataLoader
from config_loader import load_config
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector


def predict(model_path, problem_type_number):
    problem_type = None
    if problem_type_number == "1":
        problem_type = "classification"
        print(f"Running classification predictions using model: {model_path}")
    elif problem_type_number == "2":
        problem_type = "regression"
        print(f"Running regression predictions using model: {model_path}")
    else:
        print("Invalid problem type. Exiting.")
        exit(1)

    # Load the test database file
    input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    db_dir = os.path.join(input_folder, 'test.db')
    data = DataLoader()._load_from_sqlite(db_dir)

    # Process data
    cleaned_data = DataCleaner().clean_data(data)
    data_selected = FeatureSelector(problem_type).select_features(cleaned_data)
    data_engineered = FeatureEngineer().engineer_features(data_selected)
    preprocessed_data = DataPreprocessor(problem_type).preprocess(data_engineered)

    # Load the model
    with open("models/CLASSIFICATION_linear_regression.pkl", "rb") as f:
        model = joblib.load(f)
        
    predictions = model.predict(preprocessed_data)
    
    # Save predictions to results folder
    results_dir = load_config()['paths']['results_dir']
    predictions.to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)
    print('predictions shoul dbe saved')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions using a trained model.")
    parser.add_argument("--model", required=True, help="Path to the model file.")
    parser.add_argument("--problem", required=True, help="Problem type (1 for classification, 2 for regression).")
    args = parser.parse_args()

    predict(args.model, args.problem)
