import torch 
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models import CNN
from compute_score import compute_score, subset_classes

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                    ])

trainset = MNIST(root="../../data/mnist", train=True, download=True, transform=transform)
dataset_classes = subset_classes(trainset, samples_per_class=30, device=DEVICE)

inputs = []
targets = []
for c in dataset_classes:
    for data in dataset_classes[c]:
        inputs.append(data)
        targets.append(c)
inputs = torch.stack(inputs).to(DEVICE)
targets = torch.tensor(targets).to(DEVICE)


for hid_channels in [12, 32, 64, 128, 256]:

    all_losses = []
    all_borne_infinie = []
    all_borne_finie = []

    for i in range(10):
        print(f"**** model {i} ****")

        model = CNN(hidden_channels=hid_channels).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)

        records_loss = []
        records_borne_timesteps = []
        records_borne_lambda0 = []
        records_loss_exp = []

        lambda_min_init, classes = compute_score(model, dataset_classes, device=DEVICE)

        full_score_init = 0
        score_init = []

        model.eval()
        with torch.no_grad():
            for c in classes:
                score_c = torch.sum((1 - F.softmax(model(dataset_classes[c]), dim=1)[:,c])**2)
                full_score_init+=score_c
        full_score_init/=len(inputs)
        borne_init = full_score_init
        model.train()

        records_borne_timesteps.append(borne_init.item())
        records_borne_lambda0.append(borne_init.item())
        records_loss_exp.append(borne_init.item())
        
        pbar = trange(10)
        for epoch in pbar:
            model.train()

            preds = model(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch : {epoch+1} : loss {loss.item()}")

            lambda_min, classes = compute_score(model, dataset_classes, device=DEVICE)

            full_score = 0
            borne_timesteps = 0
            borne_lambda0 = 0

            model.eval()
            with torch.no_grad():
                for c in classes:
                    score_c = sum((1 - F.softmax(model(dataset_classes[c]), dim=1)[:,c])**2)
                    full_score+=score_c
            model.train()

            borne_timesteps = np.exp(-lambda_min*(epoch+1))*full_score_init.detach()
            borne_lambda0 = np.exp(-lambda_min_init*(epoch+1))*full_score_init.detach()

            full_score = full_score.item()/len(inputs)
            records_borne_timesteps.append(borne_timesteps.item())
            records_borne_lambda0.append(borne_lambda0.item())
            records_loss_exp.append(full_score)
            records_loss.append(loss.item())

        all_losses.append(records_loss_exp)
        all_borne_finie.append(records_borne_timesteps)
        all_borne_infinie.append(records_borne_lambda0)

    plt.plot(np.mean(all_losses, axis=0), c="red", label="train loss")
    plt.plot(np.mean(all_borne_finie, axis=0), c="blue", label="bound $\lambda_t$")
    plt.plot(np.mean(all_borne_infinie, axis=0), c="green", label="bound $\lambda_0$")
    plt.title('Hidden channels size : '+str(hid_channels))
    plt.legend()
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig(f'../../figures/loss_bound/loss_bound_mnist_cnn_hidden_{hid_channels}.pdf', bbox_inches='tight', format="pdf")
    plt.close()