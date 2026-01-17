import csv
import json
import os
import sys
sys.path.append('../..')

import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from torch.func import functional_call, vmap, jacrev

from data.timeseries.dataset import GlobalTempDataset
from models.transformer import Transformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


def compute_ntk(model, x):
    model.eval()  # Disable dropout
    
    parameters = dict(model.named_parameters())
    
    def fnet_single(params, x_single):
        x_single = x_single.unsqueeze(0)
        output = functional_call(model, params, x_single)
        return output.squeeze(0).flatten()
    
    jac1 = vmap(jacrev(fnet_single), (None, 0), randomness='same')(parameters, x)
    jac2 = vmap(jacrev(fnet_single), (None, 0), randomness='same')(parameters, x)
    
    # Compute NTK: J @ J^T
    ntk = sum(
        torch.einsum('naf,mbf->nm', j1.flatten(2), j2.flatten(2))
        for j1, j2 in zip(jac1.values(), jac2.values())
    )
    
    return ntk


dataset = GlobalTempDataset("../../data/timeseries/globaltemp.csv", seq_len=20)

xs_train = torch.stack([dataset[i][0] for i in range(min(100, len(dataset)))]).to(DEVICE)
ys_train = torch.stack([dataset[i][1] for i in range(min(100, len(dataset)))]).reshape(-1, 1).to(DEVICE)
x_ntk = xs_train  # to compute ntk on the training data

csv_path = "../../results/loss_bound/transformer_results.csv"
# If the CSV file exists, remove it to start fresh
if os.path.exists(csv_path):
    os.remove(csv_path)

for hid_dim in [64, 128, 256, 512, 1024]:
    print(f"********** Hidden layer size : {hid_dim} **********")

    all_losses = []
    all_borne_infinie = []
    all_borne_finie = []

    # On descend le learning rate pour rester en lazy training regime
    base_lr = 0.1
    lr = base_lr / np.sqrt(hid_dim)
    
    pbar = trange(10)
    for i in pbar:
        model = Transformer(hidden_dim=hid_dim, seq_len=20).to(DEVICE)
        
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        with torch.no_grad():
            ntk = compute_ntk(model, x_ntk)
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
            epoch_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            pbar.set_description(f'model {i+1} ; for epoch {epoch} ; current train loss is {epoch_loss}')

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

        del model  # Free up memory
        torch.cuda.empty_cache()

    # epochs = np.arange(len(all_losses[0]))

    # # Compute means and stds
    # mean_loss = np.mean(all_losses, axis=0)
    # std_loss = np.std(all_losses, axis=0)

    # mean_borne_finie = np.mean(all_borne_finie, axis=0)
    # std_borne_finie = np.std(all_borne_finie, axis=0)

    # mean_borne_infinie = np.mean(all_borne_infinie, axis=0)
    # std_borne_infinie = np.std(all_borne_infinie, axis=0)

    # # Plot curves with intervals
    # plt.plot(epochs, mean_loss, c="red", label="Train loss", linewidth=2)
    # plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color="red", alpha=0.2)

    # plt.plot(epochs, mean_borne_finie, c="blue", label=r"Bound $\lambda_t$", linewidth=2)
    # plt.fill_between(epochs, mean_borne_finie - std_borne_finie, mean_borne_finie + std_borne_finie, color="blue", alpha=0.2)

    # plt.plot(epochs, mean_borne_infinie, c="green", label=r"Bound $\lambda_0$", linewidth=2)
    # plt.fill_between(epochs, mean_borne_infinie - std_borne_infinie, mean_borne_infinie + std_borne_infinie, color="green", alpha=0.2)

    # # Labels and formatting
    # plt.xlabel("Epochs", fontsize=12)
    # plt.ylabel("Value", fontsize=12)
    # plt.title(f"Transformer - Hidden Layer Size: {hid_dim}", fontsize=14)
    # plt.legend(fontsize=10)
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f'../figures/loss_bound/loss_bound_transformer_hidden_{hid_dim}.pdf', bbox_inches='tight', format="pdf")
    # plt.close()
