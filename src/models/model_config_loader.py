import yaml
from pathlib import Path

MODEL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/model_config.yaml"

def load_model_config(a, b, c=None):
    """Load parameters from the main configuration file (model_config.yaml)."""
    with open(MODEL_CONFIG_PATH, "r") as file:
        model_config = yaml.safe_load(file)
    if c is not None:
        config_desired = model_config[a][b][c]
    else:
        config_desired = model_config[a][b]
    return config_desired

# Test the functions
print(load_model_config('evaluation_classification', 'primary_metric'))
print(load_model_config('evaluation_regression', 'primary_metric'))
