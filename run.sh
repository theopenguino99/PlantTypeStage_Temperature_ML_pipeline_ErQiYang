#!/bin/bash

# Exit script if any command fails
set -e

# Default configuration file
CONFIG_FILE="config/config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run.sh [--config CONFIG_FILE]"
      exit 1
      ;;
  esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Configuration file '$CONFIG_FILE' not found."
  exit 1
fi

echo "Starting ML pipeline with configuration: $CONFIG_FILE"

# Add src directory to PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Create necessary directories if they don't exist
mkdir -p data/processed
mkdir -p models
mkdir -p results

# Run the ML pipeline
python -m src.main --config "$CONFIG_FILE"

echo "ML pipeline completed successfully!"

# Prompt user to choose the problem type
while true; do
  echo "Select the problem type:"
  echo "1) Classification"
  echo "2) Regression"
  read -p "Enter your choice (1 or 2): " PROBLEM_TYPE

  if [ "$PROBLEM_TYPE" -eq 1 ]; then
    RESULT_FILE="results/plant_results.csv"
    echo "Classification results:"
    break
  elif [ "$PROBLEM_TYPE" -eq 2 ]; then
    RESULT_FILE="results/temperature_regression_results"
    echo "Regression results:"
    break
  else
    echo "Invalid choice. Please enter 1 or 2."
  fi
done

# Display the results
if [ "$PROBLEM_TYPE" -eq 1 ]; then
  RESULT_FILE=$(ls results | grep "^plant")
elif [ "$PROBLEM_TYPE" -eq 2 ]; then
  RESULT_FILE=$(ls results | grep "^temperature")
fi

# Prompt user to choose the model based on the problem type
echo "Available models:"

while true; do
  if [ "$PROBLEM_TYPE" -eq 2 ]; then
    MODELS=($(ls models | grep "^REGRESSION"))  # Get list of regression models
  elif [ "$PROBLEM_TYPE" -eq 1 ]; then
    MODELS=($(ls models | grep "^CLASSIFICATION"))  # Get list of classification models
  else
    echo "Invalid problem type."
    exit 1
  fi

  # Check if models exist
  if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models found for the selected problem type."
    exit 1
  fi

  # Display models with numbers
  for i in "${!MODELS[@]}"; do
    echo "$((i+1))) ${MODELS[$i]}"
  done

  # Ask user to select a model by number
  read -p "Enter the number of the model to use: " MODEL_INDEX

  # Validate user input
  if [[ "$MODEL_INDEX" =~ ^[0-9]+$ ]] && [ "$MODEL_INDEX" -ge 1 ] && [ "$MODEL_INDEX" -le "${#MODELS[@]}" ]; then
    break
  else
    echo "Invalid selection. Please enter a valid number."
  fi
done

# Assign selected model
MODEL_NAME="${MODELS[$((MODEL_INDEX-1))]}"

echo "You selected: $MODEL_NAME"

# Ensure the model file exists
if [ ! -f "models/$MODEL_NAME" ]; then
  echo "Error: Model file not found: models/$MODEL_NAME"
  exit 1
fi

# Ask the user if they want to run the prediction
while true; do
  read -p "Do you want to run the prediction? (yes/no): " RUN_PREDICTION
  case "$RUN_PREDICTION" in
    [Yy]* ) 
      echo "Proceeding to run prediction..."
      break
      ;;
    [Nn]* )
      echo "Prediction skipped."
      exit 0
      ;;
    * ) 
      echo "Please answer yes or no."
      ;;
  esac
done

# Prompt user to place the .db test file in the data directory
echo "Please ensure a test.db test file (name as test.db please!!) is placed in the 'data' directory before proceeding."
read -p "Press Enter to continue once the file is in place..."

# Run predictions
python -m src.predict --model "models/$MODEL_NAME" --problem "$PROBLEM_TYPE"
