# src/data/preprocess.py

import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from src.utils import load_config

def process_npy_folder(raw_path, processed_path, name=""):
    """
    Processes a folder of .npy floor plan arrays.
    - Loads each .npy file.
    - Converts it to a PyTorch tensor.
    - Adds a channel dimension.
    - Normalizes the tensor to be between -1 and 1.
    - Saves the processed tensor.
    """
    print(f"Processing {name} .npy files from {raw_path}...")
    
    os.makedirs(processed_path, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(raw_path, '*.npy'))
    print(f"Found {len(npy_files)} files to process.")
    
    for i, file_path in enumerate(tqdm(npy_files, desc=f"Processing {name}")):
        try:
            # 1. Load the NumPy array
            array = np.load(file_path)
            
            # 2. Convert to a PyTorch tensor
            tensor = torch.from_numpy(array).float()
            
            # 3. Add a channel dimension (e.g., from [H, W] to [1, H, W])
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            
            # 4. Normalize the tensor
            # The MSD data uses integer class labels (0, 1, 2...). We'll scale these
            # to the [-1, 1] range. Assuming max class label is around 12.
            tensor = (tensor / 12.0) * 2.0 - 1.0
            
            # 5. Save the processed tensor
            save_path = os.path.join(processed_path, f'{name}_{i}.pt')
            torch.save(tensor, save_path)
            
        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    print(f"{name} data processing complete. Files saved to {processed_path}")

if __name__ == '__main__':
    data_config = load_config('config/data_config.yaml')
    
    if data_config:
        # --- Process the MSD Train Set ---
        process_npy_folder(
            raw_path=data_config['raw_data']['msd_train'],
            processed_path=data_config['processed_data']['msd_2d_train'],
            name="msd_train"
        )

        # --- Process the MSD Test Set ---
        process_npy_folder(
            raw_path=data_config['raw_data']['msd_test'],
            processed_path=data_config['processed_data']['msd_2d_test'],
            name="msd_test"
        )