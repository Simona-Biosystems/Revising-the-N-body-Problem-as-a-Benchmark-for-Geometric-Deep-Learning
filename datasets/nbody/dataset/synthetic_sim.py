import numpy as np
from tqdm import tqdm


class GravitySim(object):
    def __init__(
        self,
        n_balls=100,
        loc_std=1,
        vel_norm=0.5,
        interaction_strength=1,
        noise_var=0,
        dt=0.001,
        softening=0.1,
        dim=3,
    ):
        self.n_balls = n_balls
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.dt = dt
        self.softening = softening

        self.dim = dim

    @staticmethod
    def compute_acceleration(pos, mass, G, softening):
        # positions r = [x,y,z] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # matrix that stores 1/r^3 for all particle pairwise particle separations
        inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

        ax = G * (dx * inv_r3) @ mass
        ay = G * (dy * inv_r3) @ mass
        az = G * (dz * inv_r3) @ mass

        # pack together the acceleration components
        a = np.hstack((ax, ay, az))
        return a

    def simulate_step(self, pos, vel, acc, mass):
        # (1/2) kick
        vel += acc * self.dt / 2.0

        # drift
        pos += vel * self.dt

        # update accelerations
        acc = self.compute_acceleration(
            pos, mass, self.interaction_strength, self.softening
        )

        # (1/2) kick
        vel += acc * self.dt / 2.0

        return pos, vel, acc

    def sample_trajectory(
        self,
        T=10000,
        sample_freq=10,
        og_pos_save=None,
        og_vel_save=None,
        og_force_save=None,
        random_seed=None,
        log_progress=False,
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed(None)

        assert T % sample_freq == 0

        T_save = int(T / sample_freq)

        N = self.n_balls

        pos_save = np.zeros((T_save, N, self.dim))
        vel_save = np.zeros((T_save, N, self.dim))
        force_save = np.zeros((T_save, N, self.dim))

        mass = np.ones((N, 1))
        if og_pos_save is None:
            # ensures that we have same density of balls in starting position as we use in 5 bodies experiment.
            std_dev = np.cbrt(N / 5)
            # Specific sim parameters
            pos = (
                np.random.randn(N, self.dim) * std_dev
            )  # randomly selected positions and velocities
            vel = np.random.randn(N, self.dim)

            # Convert to Center-of-Mass frame
            vel -= np.mean(mass * vel, 0) / np.mean(mass)

        else:
            pos = np.copy(og_pos_save[-1])
            vel = np.copy(og_vel_save[-1])

        # calculate initial gravitational accelerations
        acc = self.compute_acceleration(
            pos, mass, self.interaction_strength, self.softening
        )

        if og_pos_save is not None:
            pos, vel, acc = self.simulate_step(pos, vel, acc, mass)

        counter = 0

        # Initialize progress bar if log_progress is True
        if log_progress:
            iteration_range = tqdm(range(T), desc="Simulating trajectory")
        else:
            iteration_range = range(T)

        for i in iteration_range:
            if i % sample_freq == 0:
                pos_save[counter] = pos
                vel_save[counter] = vel
                force_save[counter] = acc * mass
                counter += 1

            pos, vel, acc = self.simulate_step(pos, vel, acc, mass)

        # Add noise to observations
        pos_save += np.random.randn(T_save, N, self.dim) * self.noise_var
        vel_save += np.random.randn(T_save, N, self.dim) * self.noise_var
        force_save += np.random.randn(T_save, N, self.dim) * self.noise_var

        if og_pos_save is not None:
            pos_save = np.concatenate((og_pos_save, pos_save), axis=0)
            vel_save = np.concatenate((og_vel_save, vel_save), axis=0)
            force_save = np.concatenate((og_force_save, force_save), axis=0)

        return pos_save, vel_save, force_save, mass

    def _energy(self, pos, vel, mass, G):
        # Kinetic Energy:
        KE = 0.5 * np.sum(np.sum(mass * vel**2))

        # Potential Energy:

        # positions r = [x,y,z] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # matrix that stores 1/r for all particle pairwise particle separations
        inv_r = np.sqrt(dx**2 + dy**2 + dz**2 + self.softening**2)
        inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

        # sum over upper triangle, to count each interaction only once
        PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

        return KE, PE, KE + PE
