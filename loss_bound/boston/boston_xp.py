import csv
import json
import os
import sys
sys.path.append('../../')

import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from data.boston.dataset import BostonDataset
from models.dense import NN
from compute_ntk import compute_ntk

dataset = BostonDataset("../../data/boston/Boston.csv")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

xs_train = torch.stack([trainset[i][0] for i in range(100)])
ys_train = torch.stack([trainset[i][1] for i in range(100)]).reshape(-1, 1)
x_ntk = xs_train[:100]  # to compute ntk on a subset to save time

csv_path = "../../results/loss_bound/boston_results.csv"
# If the CSV file exists, remove it to start fresh
if os.path.exists(csv_path):
    os.remove(csv_path)

for hid_dim in [10, 100, 1000]:
    print(f"********** Hidden layer size : {hid_dim} **********")

    all_losses = []
    all_borne_infinie = []
    all_borne_finie = []

    pbar = trange(10)
    for i in pbar:
        model = NN(in_dim=13, hidden_dim=hid_dim, out_dim=1)
        
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        with torch.no_grad():
            ntk = compute_ntk(model,x_ntk)
            model.eval()
            preds = model(xs_train)
            model.train()
        l, v = torch.linalg.eig(ntk)
        init_error = criterion(preds, ys_train).item()
        losses_simu = [init_error]
        lambdas_simu = [torch.min(l.real).item()]
        
        for epoch in range(50):
            preds = model(xs_train)
            loss = criterion(preds, ys_train)
            epoch_loss=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                ntk = compute_ntk(model, x_ntk)
                model.eval()
                preds = model(xs_train)
                model.train()
            l, v = torch.linalg.eig(ntk)
            train_loss = criterion(preds, ys_train)
            lam_min = torch.min(l.real)
            losses_simu.append(train_loss.item())
            lambdas_simu.append(lam_min.item())
            pbar.set_description(f'model {i+1} ; for epoch {epoch} ; current train loss is {epoch_loss/len(trainset)}')

        borne_infin = np.exp(-lambdas_simu[0]*(np.linspace(1, len(lambdas_simu), len(lambdas_simu))-1))*init_error
        borne_fin = np.exp(-np.asarray(lambdas_simu)*(np.linspace(1, len(lambdas_simu), len(lambdas_simu))-1))*init_error

        all_losses.append(losses_simu)
        all_borne_finie.append(borne_fin)
        all_borne_infinie.append(borne_infin)

        # Write this iteration's results to CSV
        row = {
            "hidden_dim": json.dumps(hid_dim),
            "iteration": json.dumps(i),
            "losses": json.dumps(losses_simu),
            "borne_finie": json.dumps(borne_fin.tolist()),
            "borne_infinie": json.dumps(borne_infin.tolist()),
        }
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["hidden_dim", "iteration", "losses", "borne_finie", "borne_infinie"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    epochs = np.arange(len(all_losses[0]))

    # Compute means and stds
    mean_loss = np.mean(all_losses, axis=0)
    std_loss = np.std(all_losses, axis=0)

    mean_borne_finie = np.mean(all_borne_finie, axis=0)
    std_borne_finie = np.std(all_borne_finie, axis=0)

    mean_borne_infinie = np.mean(all_borne_infinie, axis=0)
    std_borne_infinie = np.std(all_borne_infinie, axis=0)

    # Plot curves with intervals
    plt.plot(epochs, mean_loss, c="red", label="Train loss", linewidth=2)
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color="red", alpha=0.2)

    plt.plot(epochs, mean_borne_finie, c="blue", label=r"Bound $\lambda_t$", linewidth=2)
    plt.fill_between(epochs, mean_borne_finie - std_borne_finie, mean_borne_finie + std_borne_finie, color="blue", alpha=0.2)

    plt.plot(epochs, mean_borne_infinie, c="green", label=r"Bound $\lambda_0$", linewidth=2)
    plt.fill_between(epochs, mean_borne_infinie - std_borne_infinie, mean_borne_infinie + std_borne_infinie, color="green", alpha=0.2)

    # Labels and formatting
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title(f"Hidden Layer Size: {hid_dim}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../../figures/loss_bound/loss_bound_boston_hidden_{hid_dim}.pdf', bbox_inches='tight', format="pdf")
    plt.close()