import sys
sys.path.append('../')

import torch 
import torch.nn as nn 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from constant.models.classification import NN, CNN
from compute_ntk import get_ntk, get_fnet_single

DEVICE="cpu"

transform = Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                    ])

dataset = MNIST("../data/mnist/", download=True, train=True, transform=transform)

x_ntk = torch.stack([dataset[i][0] for i in range(20)]).to(DEVICE)

# Fully connected network hidden dimensions to test

ITER=50

results_dict = {
    50:[[] for _ in range(ITER)],
    100:[[] for _ in range(ITER)],
    500:[[] for _ in range(ITER)],
    1_000:[[] for _ in range(ITER)],
    5_000:[[] for _ in range(ITER)],
    10_000:[[] for _ in range(ITER)],
}

pairwise_dists = {}  # store all distances per dim

for dim in results_dict.keys():

    ntk_vectors = []

    # Compute NTK vectors for each iteration
    for iter in range(ITER):
        model = NN(dim).to(DEVICE)
        parameters = {k: v.detach() for k, v in model.named_parameters()}
        fnet_single = get_fnet_single(model)

        ntk = get_ntk(fnet_single, parameters, x_ntk, multi=True)
        n = ntk.shape[0]
        idx = torch.triu_indices(n, n)
        ntk_flat = ntk[idx[0], idx[1]].cpu().numpy()

        ntk_vectors.append(ntk_flat)
        results_dict[dim][iter] = ntk_flat

    # Compute pairwise distances between all 50 flattened NTKs
    pairwise = []
    for i, j in combinations(range(ITER), 2):
        diff = ntk_vectors[i] - ntk_vectors[j]
        dist = np.linalg.norm(diff)
        pairwise.append(dist)

    pairwise_dists[dim] = np.array(pairwise)

# Print mean pairwise distance for each dimension
for dim, dists in pairwise_dists.items():
    print(f"Dim {dim}: mean pairwise NTK distance = {dists.mean():.4f}")

# ==========================
# 📊 Plot NTK pairwise distance distributions
# ==========================

dims = sorted(pairwise_dists.keys())
data = [pairwise_dists[dim] for dim in dims]

plt.figure(figsize=(9, 5))
plt.boxplot(data, labels=dims, showmeans=True, patch_artist=True)

plt.xlabel("Hidden layer dimension", fontsize=12)
plt.ylabel("Pairwise NTK norm", fontsize=12)
plt.title("Distribution of Pairwise NTK Norms per Hidden Dimension", fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('../figures/deterministic/deterministic_mnist_ntk_pairwise_distance_boxplot_fcnn.pdf', bbox_inches='tight', format="pdf")
plt.close()

# Convolutional network hidden dimensions to test

ITER=50

results_dict = {
    16:[[0] for _ in range(ITER)],
    32:[[0] for _ in range(ITER)],
    64:[[0] for _ in range(ITER)],
    128:[[0] for _ in range(ITER)],
    256:[[0] for _ in range(ITER)],
}

pairwise_dists = {}  # store all distances per dim

for dim in results_dict.keys():

    ntk_vectors = []

    # Compute NTK vectors for each iteration
    for iter in range(ITER):
        model = CNN(dim).to(DEVICE)
        parameters = {k: v.detach() for k, v in model.named_parameters()}
        fnet_single = get_fnet_single(model)

        ntk = get_ntk(fnet_single, parameters, x_ntk, multi=True)
        n = ntk.shape[0]
        idx = torch.triu_indices(n, n)
        ntk_flat = ntk[idx[0], idx[1]].cpu().numpy()

        ntk_vectors.append(ntk_flat)
        results_dict[dim][iter] = ntk_flat

    # Compute pairwise distances between all 50 flattened NTKs
    pairwise = []
    for i, j in combinations(range(ITER), 2):
        diff = ntk_vectors[i] - ntk_vectors[j]
        dist = np.linalg.norm(diff)
        pairwise.append(dist)

    pairwise_dists[dim] = np.array(pairwise)

# Print mean pairwise distance for each dimension
for dim, dists in pairwise_dists.items():
    print(f"Dim {dim}: mean pairwise NTK distance = {dists.mean():.4f}")

# ==========================
# 📊 Plot NTK pairwise distance distributions
# ==========================

dims = sorted(pairwise_dists.keys())
data = [pairwise_dists[dim] for dim in dims]

plt.figure(figsize=(9, 5))
plt.boxplot(data, labels=dims, showmeans=True, patch_artist=True)

plt.xlabel("Hidden layer dimension", fontsize=12)
plt.ylabel("Pairwise NTK norm", fontsize=12)
plt.title("Distribution of Pairwise NTK Norms per Hidden Dimension", fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('../figures/deterministic/deterministic_mnist_ntk_pairwise_distance_boxplot_cnn.pdf', bbox_inches='tight', format="pdf")
plt.close()