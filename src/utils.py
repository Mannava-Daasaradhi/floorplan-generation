# src/utils.py

import yaml

def load_config(config_path='config/model_config.yaml'):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None

if __name__ == '__main__':
    # Example of how to use it
    model_config = load_config()
    data_config = load_config('config/data_config.yaml')
    
    if model_config and data_config:
        print("--- Model Config ---")
        print(model_config)
        print("\n--- Data Config ---")
        print(data_config)