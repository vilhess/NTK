import sys
sys.path.append('../../')

import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from data.boston.dataset import BostonDataset
from model import NN
from compute_ntk import compute_ntk

dataset = BostonDataset("../../data/boston/Boston.csv")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

xs_train = torch.stack([trainset[i][0] for i in range(100)])
ys_train = torch.stack([trainset[i][1] for i in range(100)]).reshape(-1, 1)
x_ntk = xs_train[:100]  # to compute ntk on a subset to save time

# hidden layer size 10 

all_losses_10 = []
all_borne_infinie_10 = []
all_borne_finie_10 = []

pbar = trange(10)
for i in pbar:
    model = NN(in_dim=13, hidden_dim=10)
    
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

    all_losses_10.append(losses_simu)
    all_borne_finie_10.append(borne_fin)
    all_borne_infinie_10.append(borne_infin)

plt.plot(np.mean(all_losses_10, axis=0), c="red", label="train loss")
plt.plot(np.mean(all_borne_finie_10, axis=0), c="blue", label="borne $\lambda_t$")
plt.plot(np.mean(all_borne_infinie_10, axis=0), c="green", label="borne $\lambda_0$")
plt.title('Hidden layer size : 10')
plt.legend()
plt.tight_layout()
plt.savefig('../../figures/loss_bound/loss_bound_boston_hidden_10_2.pdf', bbox_inches='tight', format="pdf")
plt.close()

# hidden layer size 100

all_losses_100 = []
all_borne_infinie_100 = []
all_borne_finie_100 = []

pbar = trange(10)
for i in pbar:
    model = NN(in_dim=13, hidden_dim=100)
    
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

    all_losses_100.append(losses_simu)
    all_borne_finie_100.append(borne_fin)
    all_borne_infinie_100.append(borne_infin)

plt.plot(np.mean(all_losses_100, axis=0), c="red", label="train loss")
plt.plot(np.mean(all_borne_finie_100, axis=0), c="blue", label="borne $\lambda_t$")
plt.plot(np.mean(all_borne_infinie_100, axis=0), c="green", label="borne $\lambda_0$")
plt.title('Hidden layer size : 100')
plt.legend()
plt.tight_layout()
plt.savefig('../../figures/loss_bound/loss_bound_boston_hidden_100_2.pdf', bbox_inches='tight', format="pdf")
plt.close()

# hidden layer size 1000

all_losses_1000 = []
all_borne_infinie_1000 = []
all_borne_finie_1000 = []

pbar = trange(10)
for i in pbar:
    model = NN(in_dim=13, hidden_dim=1000)
    
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

    all_losses_1000.append(losses_simu)
    all_borne_finie_1000.append(borne_fin)
    all_borne_infinie_1000.append(borne_infin)

plt.plot(np.mean(all_losses_1000, axis=0), c="red", label="train loss")
plt.plot(np.mean(all_borne_finie_1000, axis=0), c="blue", label="borne $\lambda_t$")
plt.plot(np.mean(all_borne_infinie_1000, axis=0), c="green", label="borne $\lambda_0$")
plt.title('Hidden layer size : 1000')
plt.legend()
plt.xlabel('Epochs')
plt.tight_layout()
plt.savefig('../../figures/loss_bound/loss_bound_boston_hidden_1000_2.pdf', bbox_inches='tight', format="pdf")
plt.close()