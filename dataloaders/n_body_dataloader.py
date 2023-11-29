from abc import abstractmethod

from dataloaders.base_dataloader import BaseDataLoader
from datasets.nbody.dataset_gravity_otf import GravityDatasetOtf


class NBodyDataLoader(BaseDataLoader):
    def __init__(self, args):
        super().__init__(args)

    def create_dataset(self):
        return GravityDatasetOtf(
            dataset_name=self.args.dataset_name,
            num_nodes=self.args.num_atoms,
            target=self.args.target,
            sample_freq=self.args.sample_freq,
            batch_size=self.args.batch_size,
            double_precision=self.args.double_precision,
            center_of_mass=self.args.center_of_mass,
            use_cached=self.args.model_path is None,
            cache_data=True,
        )

    def get_batch(self):
        batch_data = next(self.dataset_iter)

        data = [d.view(-1, d.size(2)) for d in batch_data]
        data = [d.double() if self.args.double_precision else d.float() for d in data]
        data = [d.to(self.device) for d in data]
        return data, None

    @abstractmethod
    def preprocess_batch(self, data, training=True):
        pass
