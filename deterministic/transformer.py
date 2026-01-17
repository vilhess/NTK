import csv
import json
import os
import sys
sys.path.append('../')
from figures.figure_setup import configure_seaborn
from models.transformer import Transformer
import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from compute_ntk import get_ntk, get_fnet_single
from data.timeseries.dataset import GlobalTempDataset

DEVICE = "cpu"

dataset = GlobalTempDataset(path="../data/timeseries/globaltemp.csv", seq_len=10)

x_ntk = torch.stack([dataset[i][0] for i in range(min(100, len(dataset)))]).to(DEVICE)

ITER = 50

csv_path = "../results/deterministic/transformer_results.csv"
# If the CSV file exists, remove it to start fresh
if os.path.exists(csv_path):
    os.remove(csv_path)

results_dict = {
    64: [[0] for _ in range(ITER)],
    128: [[0] for _ in range(ITER)],
    256: [[0] for _ in range(ITER)],
    512: [[0] for _ in range(ITER)],
    1024: [[0] for _ in range(ITER)],
}

pairwise_dists = {}  # store all distances per dim

for dim in results_dict.keys():
    print(f"Processing hidden_dim={dim}...")

    ntk_vectors = []

    # Compute NTK vectors for each iteration
    for iter in range(ITER):
        model = Transformer(hidden_dim=dim, seq_len=10).to(DEVICE)
        parameters = {k: v.detach() for k, v in model.named_parameters()}
        fnet_single = get_fnet_single(model)

        ntk = get_ntk(fnet_single, parameters, x_ntk, multi=False)
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

    # Write results for this dimension to CSV
    row = {
        "hidden_dim": json.dumps(dim),
        "mean_pairwise_dist": json.dumps(float(pairwise_dists[dim].mean())),
        "std_pairwise_dist": json.dumps(float(pairwise_dists[dim].std())),
        "pairwise_dists": json.dumps(pairwise_dists[dim].tolist()),
    }
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hidden_dim", "mean_pairwise_dist", "std_pairwise_dist", "pairwise_dists"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Print mean pairwise distance for each dimension
for dim, dists in pairwise_dists.items():
    print(f"Dim {dim}: mean pairwise NTK distance = {dists.mean():.4f}")

# ==========================
# 📊 Plot NTK pairwise distance distributions
# ==========================

configure_seaborn()

dims = sorted(pairwise_dists.keys())
data = [pairwise_dists[dim] for dim in dims]

plt.figure(figsize=(9, 5))
plt.boxplot(data, labels=dims, showmeans=True, patch_artist=True)

plt.xlabel("Hidden layer dimension", fontsize=12)
plt.ylabel("Pairwise NTK norm", fontsize=12)
plt.title("Distribution of Pairwise NTK Norms per Hidden Dimension (Transformer)", fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('../figures/deterministic/deterministic_transformer_ntk_pairwise_distance_boxplot.pdf', bbox_inches='tight', format="pdf")
plt.close()
