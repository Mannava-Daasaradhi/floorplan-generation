# src/data/dataset.py

import os
import torch
from torch.utils.data import Dataset
from glob import glob

class FloorplanDataset(Dataset):
    """
    Custom PyTorch Dataset for loading preprocessed floor plan tensors.
    """
    def __init__(self, processed_dir):
        """
        Args:
            processed_dir (str): Directory containing the processed .pt files.
        """
        self.file_paths = glob(os.path.join(processed_dir, '*.pt'))
        if not self.file_paths:
            raise RuntimeError(f"No .pt files found in {processed_dir}. Did you run the preprocessing script?")
        print(f"Found {len(self.file_paths)} processed floor plans.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        """
        # Load the preprocessed tensor from disk
        tensor = torch.load(self.file_paths[idx])
        return tensor

if __name__ == '__main__':
    # Example of how to use the dataset
    # This assumes you have run the preprocessing script first
    dataset = FloorplanDataset(processed_dir='data/processed/rplan_2d/')
    
    # Get the first sample
    if len(dataset) > 0:
        first_sample = dataset[0]
        print("Successfully loaded a sample.")
        print("Sample shape:", first_sample.shape)
        print("Sample data type:", first_sample.dtype)