import os
from pathlib import Path
import yaml

# Loading directory of the preprocessing config file
def load_preprocessing_config(self):
        """
        Load preprocessing configuration.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Preprocessing configuration
        """
        dir = os.path.join(Path(__file__).resolve().parents[2], self.config['pipeline']['preprocessing_config'])
        print('The preprocessing config file is in ', dir)
        with open(dir, 'r') as file:
            return yaml.safe_load(file)