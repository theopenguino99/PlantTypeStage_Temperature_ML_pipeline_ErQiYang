import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/config.yaml"
PREPROCESSING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/preprocessing_config.yaml"

def load_config():
    """Load the main configuration file (config.yaml)."""
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)
    

def load_preprocessing_config():
    """Load the preprocessing configuration from config.yaml."""
    config = load_config()
    preprocessing_path = CONFIG_PATH.parent.parent / config['pipeline']['preprocessing_config']
    with open(preprocessing_path, "r") as file:
        return yaml.safe_load(file)

def load_raw_data():
    """Load the raw_data directory (a .db SQL data base)."""
    config = load_config()
    preprocessing_path = CONFIG_PATH.parent.parent / config['paths']['raw_data']
    return preprocessing_path