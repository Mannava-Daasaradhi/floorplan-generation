# src/train.py

import argparse
import torch
from torch.utils.data import DataLoader
from src.utils import load_config
from src.models import floorplan_gan # Example import
from src.data.dataset import FloorplanDataset # <-- IMPORT THE NEW DATASET

def train(model_name, dataset_name):
    """
    Main training loop.
    
    Args:
        model_name (str): The name of the model to train (e.g., 'floorplan_gan').
        dataset_name (str): The name of the dataset to use (e.g., 'rplan_2d').
    """
    print(f"--- Starting Training ---")
    print(f"Model: {model_name} | Dataset: {dataset_name}")

    # 1. Load configurations
    model_config = load_config('config/model_config.yaml')[model_name]
    data_config = load_config('config/data_config.yaml')

    # 2. Initialize the Dataset and DataLoader
    processed_data_path = data_config['processed_data'][dataset_name]
    dataset = FloorplanDataset(processed_dir=processed_data_path)
    data_loader = DataLoader(
        dataset,
        batch_size=model_config['batch_size'],
        shuffle=True,
        num_workers=4 # Use multiple threads to load data
    )

    # 3. Initialize the model
    if model_name == 'floorplan_gan':
        generator = floorplan_gan.Generator(
            latent_dim=model_config['latent_dim'],
            image_size=model_config['image_size']
        )
        # ... and so on
    
    # 4. Start the training loop
    print(f"Starting training loop with {len(dataset)} images...")
    # Example of iterating through the data
    for epoch in range(model_config['epochs']):
        for i, real_images in enumerate(data_loader):
            # Your training logic will go here
            # 'real_images' is a batch of tensors from the dataset
            if i == 0 and epoch == 0:
                print(f"Epoch [{epoch+1}/{model_config['epochs']}], Batch {i+1}, Batch Shape: {real_images.shape}")
                # We break here for now, as the training logic is not implemented yet
                break
        if epoch == 0:
            break
            
    print("--- Training Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a generative model for floor plan generation.")
    parser.add_argument('--model', type=str, required=True, help="Name of the model to train (from model_config.yaml).")
    # Updated to match the key in data_config.yaml
    parser.add_argument('--dataset', type=str, required=True, help="Name of the processed dataset to use (e.g. 'rplan_2d').")
    
    args = parser.parse_args()
    
    train(args.model, args.dataset)