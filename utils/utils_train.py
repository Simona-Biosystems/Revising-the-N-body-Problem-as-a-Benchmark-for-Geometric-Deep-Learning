from dataloaders.ponita_n_body_dataloader import PonitaNBodyDataLoader
from models.ponita.ponita_nbody import PONITA_NBODY
from utils.nbody_utils import get_device


def create_model(args):
    return PONITA_NBODY(
        layers=args.num_layers,
        hidden_dim=args.hidden_features,
        lr=args.lr,
    )


def create_dataloader(args):
    return PonitaNBodyDataLoader(args)
