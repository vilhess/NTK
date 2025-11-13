import sys
sys.path.append('../')

import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models.classification import NN, CNN
from compute_ntk import get_ntk, get_fnet_single
from utils import get_relative_norm

DEVICE="cuda"

transform = Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                    ])

dataset = MNIST("../data/mnist/", download=True, train=True, transform=transform)

x_ntk = torch.stack([dataset[i][0] for i in range(35)]).to(DEVICE)
target = torch.tensor([dataset[i][1] for i in range(35)]).to(DEVICE)

# Linear Fully Connected Network

EPOCHS=20
ITER=5

results_dict = {
    250:[[0] for _ in range(ITER)],
    500:[[0] for _ in range(ITER)],
    1000:[[0] for _ in range(ITER)],
    2500:[[0] for _ in range(ITER)],
    5000:[[0] for _ in range(ITER)],
}

for dim in results_dict.keys():
    for iter in range(ITER):
        print(f'*** Working on model {dim} , iter {iter+1}/{ITER} ***')

        model = NN(dim).to(DEVICE)
        parameters = {k:v.detach() for k, v in model.named_parameters()}

        fnet_single = get_fnet_single(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-5)

        ntk_init = get_ntk(fnet_single, parameters, x_ntk, multi=True)

        pbar = trange(EPOCHS)

        for epoch in pbar:
            pred = model(x_ntk)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss=loss.item()
            pbar.set_description(f"for epoch {epoch} ; training loss : {epoch_loss}")

            parameters = {k:v.detach() for k, v in model.named_parameters()}
            with torch.no_grad():
                ntk = get_ntk(fnet_single, parameters, x_ntk, multi=True)
                rel_norm = get_relative_norm(ntk, ntk_init)
            results_dict[dim][iter].append(rel_norm)

for dim, values in results_dict.items():
    # Compute mean and standard deviation over runs
    mean_results = np.mean(values, axis=0)
    std_results = np.std(values, axis=0)

    epochs = np.arange(len(mean_results))

    # Plot mean curve
    plt.plot(epochs, mean_results, label=f"{dim}", linewidth=2)

    # Plot shaded interval (±1 std)
    plt.fill_between(epochs,
                     mean_results - std_results,
                     mean_results + std_results,
                     alpha=0.2)

# Axis labels and title
plt.xlabel('Epochs', fontsize=12)
plt.ylabel(r'$\frac{\|\theta_t - \theta_0\|^2}{\|\theta_0\|^2}$', fontsize=14)
plt.title("Evolution of the Relative NTK Norm During Training", fontsize=14)

# Set x-ticks explicitly for all epochs
plt.xticks(epochs)

# Legend and grid
plt.legend(title='Width', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/constant/constant_mnist_ntk_relative_norm_fcnn.pdf', bbox_inches='tight', format="pdf")
plt.close()

# Convolutional Neural Network

EPOCHS=10
ITER=5

results_dict = {
    16:[[0] for _ in range(ITER)],
    32:[[0] for _ in range(ITER)],
    64:[[0] for _ in range(ITER)],
    128:[[0] for _ in range(ITER)],
    256:[[0] for _ in range(ITER)],
    512:[[0] for _ in range(ITER)],
}

for dim in results_dict.keys():
    for iter in range(ITER):
        print(f'*** Working on model {dim} , iter {iter+1}/{ITER} ***')

        model = CNN(dim).to(DEVICE)
        parameters = {k:v.detach() for k, v in model.named_parameters()}

        fnet_single = get_fnet_single(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-5)

        ntk_init = get_ntk(fnet_single, parameters, x_ntk, multi=True)

        pbar = trange(EPOCHS)
        for epoch in pbar:
            epoch_loss = 0
            pred = model(x_ntk)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss=loss.item()

            pbar.set_description(f"for epoch {epoch} ; training loss : {epoch_loss}")       
            parameters = {k:v.detach() for k, v in model.named_parameters()}
            with torch.no_grad():
                ntk = get_ntk(fnet_single, parameters, x_ntk, multi=True)
                rel_norm = get_relative_norm(ntk, ntk_init)
            results_dict[dim][iter].append(rel_norm)

for dim, values in results_dict.items():
    # Compute mean and std over runs
    mean_results = np.mean(values, axis=0)
    std_results = np.std(values, axis=0)

    epochs = np.arange(len(mean_results))

    # Plot mean curve
    plt.plot(epochs, mean_results, label=f"{dim} out-channels", linewidth=2)

    # Fill standard deviation interval
    plt.fill_between(epochs,
                     mean_results - std_results,
                     mean_results + std_results,
                     alpha=0.2)

# Labels and formatting
plt.xlabel('Epochs', fontsize=12)
plt.ylabel(r'$\frac{\|\theta_t - \theta_0\|^2}{\|\theta_0\|^2}$', fontsize=14)
plt.title("Evolution of the Relative NTK Norm During Training", fontsize=14)

# X-axis ticks for all epochs
plt.xticks(epochs)

plt.legend(title='Model Width', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/constant/constant_mnist_ntk_relative_norm_cnn.pdf', bbox_inches='tight', format="pdf")
plt.close()