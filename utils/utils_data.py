# model
# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

# data pre-processing
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from ase import Atoms
from ase.visualize.plot import plot_atoms
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

# utilities
from tqdm import tqdm

# format progress bar
bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
tqdm.pandas(bar_format=bar_format)


# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["font.size"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = textsize


# colors for datasets
palette = ["#2876B2", "#F39957", "#67C7C2", "#C86646"]
datasets = ["train", "valid", "test"]
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", [palette[k] for k in [0, 2, 1]]
)


def calculate_energies(pos, vel, mass, bodies, G, softening):
    arr = []
    for i in range(int(len(pos) / bodies)):
        arr.append(
            _calculate_energies(
                pos[(i * bodies) : (i * bodies + bodies)],
                vel[(i * bodies) : (i * bodies + bodies)],
                mass[(i * bodies) : (i * bodies + bodies)],
                G,
                softening,
            )
        )
    return np.array(arr)


def _calculate_energies(pos, vel, mass, G, softening):
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
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2 + softening**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE
