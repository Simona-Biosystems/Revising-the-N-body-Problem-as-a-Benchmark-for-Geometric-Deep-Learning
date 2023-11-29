import torch
from torch_geometric.data import Data

from dataloaders.n_body_dataloader import NBodyDataLoader
from utils.build_fully_connected_graph import build_graph_with_knn


class PonitaNBodyDataLoader(NBodyDataLoader):
    def __init__(self, args):
        super().__init__(args)

    def create_dataset(self):
        return super().create_dataset()

    def get_batch(self):
        return super().get_batch()

    def preprocess_batch(self, data, training=True):
        if training:
            loc, vel, _, mass, y = data
        else:
            loc, vel, _, mass = data

        graph = Data(torch.hstack([mass]))
        graph.pos = loc
        graph.vec = vel
        graph.vec = graph.vec.reshape(graph.vec.shape[0], 1, graph.vec.shape[1])
        graph.mass = mass
        edge_index = build_graph_with_knn(
            loc,
            self.dataset.batch_size,
            self.dataset.num_nodes,
            self.device,
            self.args.num_neighbors,
        )
        graph.edge_index = edge_index
        if training:
            graph.y = y
        graph.to(self.device)
        return graph
