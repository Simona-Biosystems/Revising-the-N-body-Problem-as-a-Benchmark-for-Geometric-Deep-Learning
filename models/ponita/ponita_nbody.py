import pytorch_lightning as pl
import torch
import torchmetrics

from models.ponita.models.ponita_pg import PonitaFiberBundle
from models.ponita.transforms.random_rotate import RandomRotate


class PONITA_NBODY(pl.LightningModule):
    """Graph Neural Network module"""

    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-5,
        warmup=10,
        layer_scale=1e-6,
        train_augm=False,
        hidden_dim=64,
        layers=4,
        radius=None,
        num_ori=20,
        basis_dim=128,
        degree=3,
        widening_factor=4,
        multiple_readouts=True,
    ):
        super().__init__()

        # Store all arguments as attributes
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.layer_scale = layer_scale
        self.train_augm = train_augm
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.radius = radius
        self.num_ori = num_ori
        self.basis_dim = basis_dim
        self.degree = degree
        self.widening_factor = widening_factor
        self.multiple_readouts = multiple_readouts

        if layer_scale == 0.0:
            layer_scale = None

        # For rotation augmentations during training and testing
        self.rotation_transform = RandomRotate(["pos", "vec", "y"], n=3)

        # The metrics to log
        self.train_metric = torchmetrics.MeanSquaredError()
        self.valid_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # Input/output specifications:
        in_channels_scalar = 1  # Mass
        in_channels_vec = 1  # Velocity
        out_channels_scalar = 0  # None
        out_channels_vec = 2  # Change of positions, velocities

        # Make the model
        self.model = PonitaFiberBundle(
            in_channels_scalar + in_channels_vec,
            hidden_dim,
            out_channels_scalar,
            layers,
            output_dim_vec=out_channels_vec,
            radius=radius,
            num_ori=num_ori,
            basis_dim=basis_dim,
            degree=degree,
            widening_factor=widening_factor,
            layer_scale=layer_scale,
            task_level="node",
            multiple_readouts=multiple_readouts,
        )

    def forward(self, graph):
        _, pred = self.model(graph)
        pos_change = pred[..., 0, :]
        vel = pred[..., 1, :]
        res = torch.hstack([pos_change, vel])
        return res

    def training_step(self, graph):
        if self.train_augm:
            graph = self.rotation_transform(graph)
        pos_pred = self(graph)
        loss = torch.mean((pos_pred - graph.y) ** 2)
        self.train_metric(pos_pred, graph.y)
        return loss

    def on_train_epoch_end(self):
        self.log("train MSE", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pos_pred = self(graph)
        self.valid_metric(pos_pred, graph.y)

    def on_validation_epoch_end(self):
        self.log("valid MSE", self.valid_metric, prog_bar=True)

    def test_step(self, graph, batch_idx):
        pos_pred = self(graph)
        self.test_metric(pos_pred, graph.y)

    def on_test_epoch_end(self):
        self.log("test MSE", self.test_metric)

    def configure_optimizers(self):
        """
        Adapted from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("layer_scale"):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr)

        return optimizer

    def get_serializable_attributes(self):
        return {
            # "num_params": sum(p.numel() for p in self.parameters()),
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup": self.warmup,
            "layer_scale": self.layer_scale,
            "train_augm": self.train_augm,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "radius": self.radius,
            "num_ori": self.num_ori,
            "basis_dim": self.basis_dim,
            "degree": self.degree,
            "widening_factor": self.widening_factor,
            "multiple_readouts": self.multiple_readouts,
        }

    def get_model_size(self):
        return self.hidden_dim
