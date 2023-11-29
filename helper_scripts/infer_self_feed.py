import argparse
import json
import os

from datasets.nbody.visualization_utils import load_dataset_from_metadata_file
from inferencer import Inferencer
from utils.nbody_utils import (
    get_dataset_metadata_path,
    get_device,
    load_model_for_inference,
)
from utils.utils_train import create_dataloader, create_model


def load_training_args(model_path):
    training_args_path = os.path.join(os.path.dirname(model_path), "training_args.json")
    with open(training_args_path, "r") as f:
        args_json = json.load(f)
    args_dict = args_json["args"]
    args = argparse.Namespace(**args_dict)
    return args


def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    args_cli = parser.parse_args()

    # Load the training arguments
    args = load_training_args(args_cli.model_path)

    # Update args with any CLI overrides
    args.model_path = args_cli.model_path

    # Set device
    device = get_device(args.gpu_id)

    # Create dataloader
    dataloader = create_dataloader(args)

    # Load the model
    model = create_model(args)
    model = load_model_for_inference(args.model_path, device)
    if args.double_precision:
        model = model.double()

    # Initialize the Inferencer
    inferencer = Inferencer(model, dataloader, args)

    # Run inference
    inferencer.run_inference(
        os.path.join(os.path.dirname(args.model_path), "trajectories_data")
    )


if __name__ == "__main__":
    main()
