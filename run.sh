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

# Create necessary directories if they don't exist
mkdir -p data/processed
mkdir -p models
mkdir -p results

# Run the ML pipeline
python -m src.main --config "$CONFIG_FILE"

echo "ML pipeline completed successfully!"
