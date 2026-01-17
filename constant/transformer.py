import csv
import json
import os
import sys
sys.path.append('../')

from figures.figure_setup import configure_seaborn

import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models.transformer import Transformer
from compute_ntk import get_ntk, get_fnet_single
from utils import get_relative_norm
from data.timeseries.dataset import GlobalTempDataset

DEVICE = "cpu"

dataset = GlobalTempDataset(path="../data/timeseries/globaltemp.csv", seq_len=10)

x_ntk = torch.stack([dataset[i][0] for i in range(min(100, len(dataset)))]).to(DEVICE)
target = torch.stack([dataset[i][1] for i in range(min(100, len(dataset)))]).to(DEVICE)

EPOCHS = 50
ITER = 50

csv_path = "../results/constant/transformer_results.csv"
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

for dim in results_dict.keys():
    for iter in range(ITER):
        print(f'*** Working on model {dim} , iter {iter+1}/{ITER} ***')

        model = Transformer(hidden_dim=dim, seq_len=10).to(DEVICE)
        parameters = {k: v.detach() for k, v in model.named_parameters()}

        fnet_single = get_fnet_single(model)
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-4)

        ntk_init = get_ntk(fnet_single, parameters, x_ntk, multi=False)

        rel_norms = [0]  # Start with 0 for epoch 0

        pbar = trange(EPOCHS)
        for epoch in pbar:
            pred = model(x_ntk)
            loss = criterion(pred.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()

            pbar.set_description(f"for epoch {epoch}/{EPOCHS} ; training loss : {epoch_loss}")

            parameters = {k: v.detach() for k, v in model.named_parameters()}
            ntk = get_ntk(fnet_single, parameters, x_ntk, multi=False)
            rel_norm = get_relative_norm(ntk, ntk_init)
            rel_norms.append(rel_norm)
            results_dict[dim][iter].append(rel_norm)

        # Write this iteration's results to CSV
        row = {
            "hidden_dim": json.dumps(dim),
            "iteration": json.dumps(iter),
            "rel_norms": json.dumps(rel_norms),
        }
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["hidden_dim", "iteration", "rel_norms"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

configure_seaborn()
plt.figure(figsize=(9, 6))

for dim, values in results_dict.items():
    # Compute mean and std over runs
    mean_results = np.mean(values, axis=0)
    std_results = np.std(values, axis=0)

    epochs = np.arange(len(mean_results))

    # Plot mean curve
    plt.plot(epochs, mean_results, label=f'{dim}', linewidth=2)

    # Fill standard deviation interval
    plt.fill_between(epochs,
                     mean_results - std_results,
                     mean_results + std_results,
                     alpha=0.2)

plt.xlabel('Epochs', fontsize=12)
plt.ylabel(r'$\frac{\|\theta_t - \theta_0\|^2}{\|\theta_0\|^2}$', fontsize=14)
plt.title("Evolution of the Relative NTK norm During Training (Transformer)", fontsize=14)
plt.legend(title='Width', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/constant/constant_transformer_ntk_relative_norm.pdf', bbox_inches='tight', format="pdf")
plt.close()
