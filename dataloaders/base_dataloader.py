from abc import ABC, abstractmethod

from utils.nbody_utils import get_device


class BaseDataLoader(ABC):
    def __init__(self, args):
        self.args = args
        self.dataset = self.create_dataset()
        self.dataset_iter = iter(self.dataset)
        self.device = get_device(self.args.gpu_id)

    @abstractmethod
    def get_batch(self):
        pass

    @abstractmethod
    def preprocess_batch(self, data, training=True):
        pass

    @abstractmethod
    def create_dataset(self):
        pass
