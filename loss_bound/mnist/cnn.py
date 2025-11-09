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
from compute_score import compute_score

DEVICE="cpu"

transform = Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                    ])

trainset = MNIST(root="../../data/mnist", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=10, pin_memory=True)

class_to_data = {i:[] for i in range(10)}

for data, label in trainset:
    class_to_data[label].append(data)

for class_label in class_to_data:
    class_to_data[class_label] = torch.stack(class_to_data[class_label]).to(DEVICE)

# hidden channel 12

all_losses_10 = []
all_borne_infinie_10 = []
all_borne_finie_10 = []

for i in range(10):
    print(f"**** model {i} ****")

    model = CNN(hidden_channels=12).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    records_loss = []
    records_borne_timesteps = []
    records_borne_lambda0 = []
    records_loss_exp = []

    lambda_min_init, classes = compute_score(model, trainset, device=DEVICE)

    full_score_init = 0
    score_init = []

    model.eval()
    for c in classes:
        score_c = torch.sum((1 - F.softmax(model(class_to_data[c]), dim=1)[:,c])**2)
        full_score_init+=score_c
    full_score_init/=len(trainset)
    borne_init = full_score_init
    model.train()

    records_borne_timesteps.append(borne_init.item())
    records_borne_lambda0.append(borne_init.item())
    records_loss_exp.append(borne_init.item())
    
    pbar = trange(10)
    for epoch in pbar:
        model.train()

        running_loss = 0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds = model(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        pbar.set_description(f"Epoch : {epoch+1} : loss {running_loss}")

        lambda_min, classes = compute_score(model, trainset, device=DEVICE)

        full_score = 0
        borne_timesteps = 0
        borne_lambda0 = 0

        model.eval()
        for c in classes:
            score_c = sum((1 - F.softmax(model(class_to_data[c]), dim=1)[:,c])**2)
            full_score+=score_c
        model.train()

        borne_timesteps = np.exp(-lambda_min*(epoch+1))*full_score_init.detach()
        borne_lambda0 = np.exp(-lambda_min_init*(epoch+1))*full_score_init.detach()

        full_score = full_score.item()/len(trainset)
        records_borne_timesteps.append(borne_timesteps.item())
        records_borne_lambda0.append(borne_lambda0.item())
        records_loss_exp.append(full_score)
        records_loss.append(running_loss)

    all_losses_10.append(records_loss_exp)
    all_borne_finie_10.append(records_borne_timesteps)
    all_borne_infinie_10.append(records_borne_lambda0)

plt.plot(np.mean(all_losses_10, axis=0), c="red", label="train loss")
plt.plot(np.mean(all_borne_finie_10, axis=0), c="blue", label="bound $\lambda_t$")
plt.plot(np.mean(all_borne_infinie_10, axis=0), c="green", label="bound $\lambda_0$")
plt.title('Hidden channels size : 12')
plt.legend()
plt.xlabel('Epochs')
plt.tight_layout()
plt.savefig('../../figures/loss_bound/loss_bound_mnist_cnn_hidden_12.pdf', bbox_inches='tight', format="pdf")
plt.close()

# hidden channel 64

all_losses_100 = []
all_borne_infinie_100 = []
all_borne_finie_100 = []

for i in range(10):
    print(f"**** model {i} ****")

    model = CNN(hidden_channels=64).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    records_loss = []
    records_borne_timesteps = []
    records_borne_lambda0 = []
    records_loss_exp = []

    lambda_min_init, classes = compute_score(model, trainset, device=DEVICE)

    full_score_init = 0
    score_init = []

    model.eval()
    for c in classes:
        score_c = torch.sum((1 - F.softmax(model(class_to_data[c]), dim=1)[:,c])**2)
        full_score_init+=score_c
    full_score_init/=len(trainset)
    borne_init = full_score_init
    model.train()

    records_borne_timesteps.append(borne_init.item())
    records_borne_lambda0.append(borne_init.item())
    records_loss_exp.append(borne_init.item())
    
    pbar = trange(10)
    for epoch in pbar:
        model.train()

        running_loss = 0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds = model(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        pbar.set_description(f"Epoch : {epoch+1} : loss {running_loss}")

        lambda_min, classes = compute_score(model, trainset, device=DEVICE)

        full_score = 0
        borne_timesteps = 0
        borne_lambda0 = 0

        model.eval()
        for c in classes:
            score_c = sum((1 - F.softmax(model(class_to_data[c]), dim=1)[:,c])**2)
            full_score+=score_c
        model.train()

        borne_timesteps = np.exp(-lambda_min*(epoch+1))*full_score_init.detach()
        borne_lambda0 = np.exp(-lambda_min_init*(epoch+1))*full_score_init.detach()

        full_score = full_score.item()/len(trainset)
        records_borne_timesteps.append(borne_timesteps.item())
        records_borne_lambda0.append(borne_lambda0.item())
        records_loss_exp.append(full_score)
        records_loss.append(running_loss)

    all_losses_100.append(records_loss_exp)
    all_borne_finie_100.append(records_borne_timesteps)
    all_borne_infinie_100.append(records_borne_lambda0)

plt.plot(np.mean(all_losses_100, axis=0), c="red", label="train loss")
plt.plot(np.mean(all_borne_finie_100, axis=0), c="blue", label="bound $\lambda_t$")
plt.plot(np.mean(all_borne_infinie_100, axis=0), c="green", label="bound $\lambda_0$")
plt.title('Hidden channel size : 64')
plt.legend()
plt.xlabel('Epochs')
plt.tight_layout()
plt.savefig('../../figures/loss_bound/loss_bound_mnist_cnn_hidden_64.pdf', bbox_inches='tight', format="pdf")
plt.close()

# hidden channel 100

all_losses_1000 = []
all_borne_infinie_1000 = []
all_borne_finie_1000 = []

for i in range(5):
    print(f"**** model {i} ****")

    model = CNN(hidden_channels=100).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    records_loss = []
    records_borne_timesteps = []
    records_borne_lambda0 = []
    records_loss_exp = []

    lambda_min_init, classes = compute_score(model, trainset, device=DEVICE)

    full_score_init = 0
    score_init = []

    model.eval()
    for c in classes:
        score_c = torch.sum((1 - F.softmax(model(class_to_data[c]), dim=1)[:,c])**2)
        full_score_init+=score_c
    full_score_init/=len(trainset)
    borne_init = full_score_init
    model.train()

    records_borne_timesteps.append(borne_init.item())
    records_borne_lambda0.append(borne_init.item())
    records_loss_exp.append(borne_init.item())
    
    pbar = trange(10)
    for epoch in pbar:
        model.train()

        running_loss = 0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds = model(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        pbar.set_description(f"Epoch : {epoch+1} : loss {running_loss}")

        model.eval()
        lambda_min, classes = compute_score(model, trainset, device=DEVICE)

        full_score = 0
        borne_timesteps = 0
        borne_lambda0 = 0

        model.eval()
        for c in classes:
            score_c = sum((1 - F.softmax(model(class_to_data[c]), dim=1)[:,c])**2)
            full_score+=score_c
        model.train()

        borne_timesteps = np.exp(-lambda_min*(epoch+1))*full_score_init.detach()
        borne_lambda0 = np.exp(-lambda_min_init*(epoch+1))*full_score_init.detach()

        full_score = full_score.item()/len(trainset)
        records_borne_timesteps.append(borne_timesteps.item())
        records_borne_lambda0.append(borne_lambda0.item())
        records_loss_exp.append(full_score)
        records_loss.append(running_loss)

    all_losses_1000.append(records_loss_exp)
    all_borne_finie_1000.append(records_borne_timesteps)
    all_borne_infinie_1000.append(records_borne_lambda0)

plt.plot(np.mean(all_losses_1000, axis=0), c="red", label="train loss")
plt.plot(np.mean(all_borne_finie_1000, axis=0), c="blue", label="bound $\lambda_t$")
plt.plot(np.mean(all_borne_infinie_1000, axis=0), c="green", label="bound $\lambda_0$")
plt.title('Hidden channel size : 100')
plt.legend()
plt.xlabel('Epochs')
plt.tight_layout()
plt.savefig('../../figures/loss_bound/loss_bound_mnist_cnn_hidden_100.pdf', bbox_inches='tight', format="pdf")
plt.close()