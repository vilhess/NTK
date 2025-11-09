import sys
sys.path.append('../')

import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models.classification import NN, CNN
from compute_ntk import get_ntk, get_fnet_single
from utils import get_relative_norm

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                    ])

dataset = MNIST("../data/mnist/", download=True, train=True, transform=transform)

x_ntk = torch.stack([dataset[i][0] for i in range(20)]).to(DEVICE)

# Linear Fully Connected Network

EPOCHS=20
ITER=5

results_dict = {
    50:[[0] for _ in range(ITER)],
    100:[[0] for _ in range(ITER)],
    500:[[0] for _ in range(ITER)],
    1000:[[0] for _ in range(ITER)],
    5000:[[0] for _ in range(ITER)],
    10000:[[0] for _ in range(ITER)],
}

for dim in results_dict.keys():
    for iter in range(ITER):
        print(f'*** Working on model {dim} , iter {iter+1}/{ITER} ***')

        model = NN(dim).to(DEVICE)
        parameters = {k:v.detach() for k, v in model.named_parameters()}
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=10)

        fnet_single = get_fnet_single(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        ntk_init = get_ntk(fnet_single, parameters, x_ntk, multi=True)

        pbar = trange(EPOCHS)

        for epoch in pbar:
            epoch_loss = 0
            for x, y in trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()

            pbar.set_description(f"for epoch {epoch} ; training loss : {epoch_loss}")

            parameters = {k:v.detach() for k, v in model.named_parameters()}
            ntk = get_ntk(fnet_single, parameters, x_ntk, multi=True)
            rel_norm = get_relative_norm(ntk, ntk_init)
            results_dict[dim][iter].append(rel_norm)

for dim in results_dict.keys():
    mean_results = np.mean(results_dict[dim], axis=0)
    plt.plot(mean_results, label=dim)

plt.xlabel('Epochs')
plt.ylabel(r'$\frac{\|\theta_t - \theta_0\|^2}{\|\theta_0\|^2}$')
plt.title("Evolution of the relative norm of the NTK during the training process")

epochs = np.arange(len(mean_results))
plt.xticks(epochs)

plt.legend()
plt.tight_layout()
plt.savefig('../figures/constant_mnist_ntk_relative_norm_fcnn.pdf', bbox_inches='tight', format="pdf")
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
}

for dim in results_dict.keys():
    for iter in range(ITER):
        print(f'*** Working on model {dim} , iter {iter+1}/{ITER} ***')

        model = CNN(dim).to(DEVICE)
        parameters = {k:v.detach() for k, v in model.named_parameters()}
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=10)

        fnet_single = get_fnet_single(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        ntk_init = get_ntk(fnet_single, parameters, x_ntk, multi=True)

        pbar = trange(EPOCHS)
        for epoch in pbar:
            epoch_loss = 0
            for x, y in trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()

            pbar.set_description(f"for epoch {epoch} ; training loss : {epoch_loss}")

            parameters = {k:v.detach() for k, v in model.named_parameters()}
            ntk = get_ntk(fnet_single, parameters, x_ntk, multi=True)
            rel_norm = get_relative_norm(ntk, ntk_init)
            results_dict[dim][iter].append(rel_norm)

for dim in results_dict.keys():
    mean_results=np.mean(results_dict[dim], axis=0)
    plt.plot(mean_results, label=f"{dim} out-channels")

plt.xlabel('Epochs')
plt.ylabel(r'$\frac{\|\theta_t - \theta_0\|^2}{\|\theta_0\|^2}$')
plt.title("Evolution of the relative norm of the NTK during the training process")
epochs = np.arange(len(mean_results))
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.savefig('../figures/constant_mnist_ntk_relative_norm_cnn.pdf', bbox_inches='tight', format="pdf")
plt.close()