import argparse
import warnings

from trainer import Trainer
from utils.utils_train import create_dataloader, create_model

warnings.filterwarnings(
    "ignore",
    message=".*The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
    module="torch.jit._check",
)


def main(args):
    model = create_model(args)

    dataloader = create_dataloader(args)

    trainer = Trainer(model, dataloader, args)

    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="N-body training script")

    # DATASET PARAMETERS
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to use (-1 for all)",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="nbody_small", help="Name of the dataset"
    )
    parser.add_argument("--num-atoms", type=int, default=5, help="Number of atoms")
    parser.add_argument(
        "--target", type=str, default="pos_dt+vel", help="Target variable"
    )
    parser.add_argument(
        "--center-of-mass", action="store_true", help="Use center of mass"
    )

    # TRAINING PARAMETERS
    parser.add_argument(
        "--test-macros-every", type=int, default=1024, help="Test macros every N steps"
    )
    parser.add_argument(
        "--save-model-every", type=int, default=10, help="Save model every N steps"
    )
    parser.add_argument(
        "--double-precision",
        action="store_true",
        default=True,
        help="Use double precision (default: True)",
    )
    parser.add_argument(
        "--no-double-precision",
        action="store_false",
        dest="double_precision",
        help="Use single precision",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a pre-trained model (optional)",
    )
    parser.add_argument("--lr-factor", type=float, default=1.0, help="LambdaLR factor")

    parser.add_argument("--energy-loss", action="store_true", help="Use energy loss")
    parser.add_argument(
        "--com-loss", action="store_true", help="Use centre of mass loss"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of wandb run, if None save_dir_path is used.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Number of training steps.",
    )

    # MODEL PARAMETERS
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=None,
        help="Number of neighbors to use for the model",
    )
    parser.add_argument(
        "--hidden-features", type=int, default=64, help="Number of hidden features"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of layers in the model"
    )

    # DATASET PARAMETERS
    parser.add_argument(
        "--sample-freq", type=int, default=10, help="Sampling frequency"
    )

    # GPU
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()
    main(args)
