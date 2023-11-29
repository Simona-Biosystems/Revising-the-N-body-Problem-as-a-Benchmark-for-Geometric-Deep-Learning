import hashlib
import json
import os
import pathlib
import pickle as pkl
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from .dataset.synthetic_sim import GravitySim


class GravityDatasetOtf:
    """
    NBodyDataset

    """

    GROUND_TRUTH_FILE_PREFIXES = ["loc", "vel", "forces", "masses"]
    DEFAULT_DATA_PATH = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "dataset", "gravity"
    )
    os.makedirs(DEFAULT_DATA_PATH, exist_ok=True)

    def __init__(
        self,
        dataset_name="nbody_small",
        target="pos_dt+vel",
        path=DEFAULT_DATA_PATH,
        batch_size=8,
        sim_length=5000,
        sample_freq=10,
        noise_var=0,
        num_nodes=5,
        vel_norm=1e-16,
        interaction_strength=2,
        dt=0.01,
        softening=0.2,
        double_precision=False,
        center_of_mass=False,
        use_cached=False,
        cache_data=True,
    ):
        self.locals = {
            k: v
            for k, v in locals().items()
            if k not in ["path", "self", "use_cached", "cache_data"]
        }
        self.cached_folder_name = self._get_cached_folder_name()
        self.data_path = "saved_simulations"
        self.noise_var = noise_var
        self.num_nodes = num_nodes
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.dt = dt
        self.softening = softening
        self.simulation: GravitySim = self.init_simulation_instance()
        self.dataset_name = dataset_name
        self.path = path
        self.double_precision = double_precision
        self.center_of_mass = center_of_mass
        self.use_cached = use_cached
        self.cache_data = cache_data

        os.makedirs(path, exist_ok=True)
        self.target = target

        self.sample_freq = sample_freq
        self.sim_length = sim_length - (sim_length % sample_freq)
        self.num_steps = sim_length // sample_freq

        self.batch_size = batch_size
        self.data_queue = []
        self.unused_indices_queue = []
        self.cache_index = 0
        if use_cached:
            self._load_saved_simulations(self.cache_index)
        else:
            self._load_more_batches()

    def get_ground_truth_trajectories(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # Generate a batch of simulations
        batch_data = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.simulation.sample_trajectory, self.sim_length, self.sample_freq
                )
                for _ in range(batch_size)
            ]
            for future in as_completed(futures):
                batch_data.append(future.result())

        metadata = self.get_serializable_attributes()
        return batch_data, metadata

    def _load_more_batches(self):
        # Preload the first batch

        future_batch, _ = self.get_ground_truth_trajectories()
        self.data_queue.append(future_batch)
        self.unused_indices_queue.append(list(range(future_batch[0][0].shape[0] - 1)))
        if self.cache_data:
            self._save_simulations(future_batch)

    def _save_simulations(self, data):
        """
        Save generated simulations to folder for later use.
        """
        simulation_caching_folder = f"{self.data_path}/{self.cached_folder_name}"
        os.makedirs(simulation_caching_folder, exist_ok=True)
        pickled_files = [
            fn for fn in os.listdir(simulation_caching_folder) if fn[-4:] == ".pkl"
        ]
        if len(pickled_files) == 0:
            file_name = "0.pkl"
        else:
            max_file_number = max([int(fn[:-4]) for fn in pickled_files])
            file_name = f"{max_file_number+1}.pkl"
        file_name = f"{simulation_caching_folder}/{file_name}"
        with open(file_name, "wb") as file:
            pkl.dump(data, file)
        print(f"simulation is saved into {file_name}")

    def _load_saved_simulations(self, index):
        """
        loads saved simulations into data_queue
        """
        print(f"Loading cached simulations from {self.cached_folder_name}")
        simulation_caching_folder = f"{self.data_path}/{self.cached_folder_name}"
        if not os.path.exists(simulation_caching_folder):
            print(f"No cached simulations found at {simulation_caching_folder}")
            self.cache_index = -1
            self._load_more_batches()
            return
        if os.path.exists(simulation_caching_folder):
            pickled_files = sorted(
                [
                    fn
                    for fn in os.listdir(simulation_caching_folder)
                    if fn[-4:] == ".pkl"
                ]
            )
            if index > len(pickled_files) - 1:
                print("Ran out of cached simulations")
                self.cache_index = -1
                self._load_more_batches()
                return

            print(f"Loading pregenerated simulation index {index}")
            file_name = pickled_files[index]
            with open(f"{simulation_caching_folder}/{file_name}", "rb") as file:
                data = pkl.load(file)
                self._push_simulations_into_data_queue(data)
            self.cache_index += 1

    def _push_simulations_into_data_queue(self, data):
        """
        Pushes data into data_queue
        """
        self.data_queue.append(data)
        self.unused_indices_queue.append(list(range(data[0][0].shape[0] - 1)))

    def _get_cached_folder_name(self):
        """
        This function uses locals to generate sha256 hash that is used as folder name
        when caching generated simulations
        """
        data_string = json.dumps(self.locals, sort_keys=True)
        hash = hashlib.sha256(data_string.encode()).hexdigest()
        return hash

    def __getitem__(self, _):
        # Load more batches
        if len(self.unused_indices_queue[0]) == 0:
            print(
                "No more unused indices in this simulation. Using next simulation batch"
            )
            self.data_queue.pop(0)
            self.unused_indices_queue.pop(0)
            if len(self.unused_indices_queue) == 0:
                print("no more simulations in queue. Loading new simulation batch")
                if self.cache_index != -1:
                    self._load_saved_simulations(self.cache_index)
                else:
                    self._load_more_batches()

        # Get the next preloaded batch
        batch_data = self.data_queue[0]
        # Extract data for the current index within the batch
        loc, vel, force, mass = zip(*batch_data)
        loc, vel, force, mass = (
            np.array(loc),
            np.array(vel),
            np.array(force),
            np.array(mass),
        )

        frame_0 = random.choice(self.unused_indices_queue[0])
        frame_T = frame_0 + 1
        self.unused_indices_queue[0].remove(frame_0)

        if self.target == "pos":
            y = loc[frame_T]
        elif self.target == "force":
            y = force[frame_T]
        elif self.target == "pos_dt+vel_dt":
            pos_dt = loc[frame_T] - loc[frame_0]  # Change in position
            vel_dt = vel[frame_T] - vel[frame_0]  # Change in velocity
            # y = torch.cat((pos_dt, vel_dt), dim=0)
            y = np.concatenate((pos_dt, vel_dt), axis=1)
        elif self.target == "pos_dt+vel":
            pos_dt = loc[:, frame_T] - loc[:, frame_0]
            y = np.concatenate((pos_dt, vel[:, frame_T]), axis=2)
        elif self.target == "pos+vel":
            selected_loc, selected_vel = loc[:, frame_T], vel[:, frame_T]
            y = np.concatenate((selected_loc, selected_vel), axis=2)
        elif self.target == "pos_com+vel":
            center_of_mass = np.mean(loc[frame_0], axis=0)
            loc_rel = loc[frame_T] - center_of_mass[None, :]
            y = np.concatenate((loc_rel, vel[frame_T]), axis=1)
        else:
            raise Exception(f"Wrong target {self.target}")

        selected_loc, selected_vel, selected_force = (
            loc[:, frame_0],
            vel[:, frame_0],
            force[:, frame_0],
        )
        return (
            torch.tensor(selected_loc),
            torch.tensor(selected_vel),
            torch.tensor(selected_force),
            torch.tensor(mass),
            torch.tensor(y),
        )

    def get_serializable_attributes(self):
        # Manually construct a dictionary of relevant attributes
        attrs = {
            "dataset_name": self.dataset_name,
            "target": self.target,
            "path": self.path,
            "batch_size": self.batch_size,
            "sim_length": self.sim_length,
            "sample_freq": self.sample_freq,
            "noise_var": self.noise_var,
            "n_balls": self.num_nodes,
            "vel_norm": self.vel_norm,
            "interaction_strength": self.interaction_strength,
            "dt": self.dt,
            "softening": self.softening,
            "double_precision": self.double_precision,
            "center_of_mass": self.center_of_mass,
        }
        return attrs

    def init_simulation_instance(self):
        return GravitySim(
            noise_var=self.noise_var,
            n_balls=self.num_nodes,
            vel_norm=self.vel_norm,
            interaction_strength=self.interaction_strength,
            dt=self.dt,
            softening=self.softening,
        )

    def simulate_one(self, sim_idx, loc, vel, mass):
        energies = []
        for i in range(loc.shape[1]):
            energy = self.simulation._energy(
                loc[sim_idx, i, :, :],
                vel[sim_idx, i, :, :],
                mass[i],
                self.simulation.interaction_strength,
            )
            energies.append(energy)
        return energies

    def get_energies_async(self, loc, vel, mass=None):
        num_simulations = loc.shape[0]

        if mass is None:
            mass = np.expand_dims(np.ones_like(loc[0, :, :, 0]), axis=-1)

        worker_partial = partial(self.simulate_one, loc=loc, vel=vel, mass=mass)

        # I/O-bound parallelization
        with ThreadPoolExecutor() as executor:
            futures = list(
                tqdm(
                    executor.map(worker_partial, range(num_simulations)),
                    total=num_simulations,
                    desc="Calculating Energies",
                )
            )

        return np.array(futures)


if __name__ == "__main__":
    dataset = GravityDatasetOtf(dataset_name="nbody")

    for item in dataset:
        loc0, vel0, force0, mass, locT = item
        print(loc0.shape, vel0.shape, force0.shape, mass.shape, locT.shape)
        break
