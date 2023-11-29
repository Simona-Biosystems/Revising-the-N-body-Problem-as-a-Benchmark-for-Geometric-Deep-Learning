import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.ponita.ponita_nbody import PONITA_NBODY


def calculate_mae(actual_data, predicted_data):
    return (abs(actual_data - predicted_data)).mean(axis=1)


def calculate_percentage_error(true_data, predicted_data):
    """
    Calculate the percentage error relative to the magnitude of the true data.

    :param true_data: numpy array of true values (either position or velocity).
    :param predicted_data: numpy array of predicted values (either position or velocity).
    :return: numpy array of percentage error values.
    """
    mae = calculate_mae(true_data, predicted_data)
    magnitude = np.linalg.norm(true_data, axis=2)
    magnitude[magnitude == 0] = np.nan  # Avoid division by zero
    return (mae / magnitude) * 100


def plot_hist_of_simulation_data(combined_data, dims=3):
    data_for_dislpay_dist = combined_data[:, :, [0, dims]]

    # Reshape the array to flatten the first two dimensions
    flattened_data = data_for_dislpay_dist.reshape(-1, data_for_dislpay_dist.shape[2])

    # Generate histograms for each feature
    for i in range(flattened_data.shape[1]):
        feature_data = flattened_data[:, i]
        histogram, bin_edges = np.histogram(feature_data, bins=50)

        plt.figure()
        plt.hist(feature_data, bins=bin_edges, alpha=0.75, color="blue")
        plt.title(f"Histogram of Feature {i + 1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def is_headless():
    return os.environ.get("DISPLAY", "") == "" and os.name != "nt"


def load_model_for_inference(model_path, device):
    print(f"Initializing model Ponita on device {device}")

    model = PONITA_NBODY()

    # Use the load_checkpoint utility function
    model = load_checkpoint(model_path, device, model=model)

    model.eval().to(device)
    return model


def load_checkpoint(model_path, device, model=None, optimizer=None, scheduler=None):
    print(f"Loading checkpoint from {model_path} on device {device}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        # If model is provided, load its state dict
        if model and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model state dict from checkpoint.")

        # Load optimizer state dict if provided and available in checkpoint
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state dict from checkpoint.")

        # Load scheduler state dict if provided and available in checkpoint
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Loaded scheduler state dict from checkpoint.")
    else:
        # If checkpoint is not a dict, assume it's a state_dict
        if model:
            model.load_state_dict(checkpoint)
            print(f"Loaded model state dict from {model_path}")
        else:
            raise ValueError(
                "Checkpoint does not contain a dictionary and no model provided."
            )

    return model


def get_device(gpu_id=0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def get_dataset_metadata_path(path):
    """
    Given a path within the run directory, finds the root of the run directory and returns the path
    to the dataset metadata file (./nbody_small_dataset/metadata.json).
    """
    import os
    import re

    # Start from the absolute path
    current_path = os.path.abspath(path)

    # Define the regex pattern for the run directory name (e.g., '2024-09-26_14-33-13')
    # Adjust the pattern if your run directories have a different naming convention
    run_dir_pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

    # Variable to store the run root directory
    run_root = None

    # Traverse up the directory tree to find the highest-level matching directory
    while True:
        dir_name = os.path.basename(current_path)
        if run_dir_pattern.match(dir_name):
            # Update run_root every time we find a matching directory
            run_root = current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:
            # Reached the filesystem root
            break
        current_path = parent

    if run_root is None:
        raise FileNotFoundError("Run directory root not found")

    # Construct the path to the metadata.json file
    metadata_path = os.path.join(run_root, "nbody_small_dataset", "metadata.json")

    return metadata_path
