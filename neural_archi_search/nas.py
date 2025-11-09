import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import sys 
sys.path.append('../')

from nas_201_api import NASBench201API as API
from xautodl.models import get_cell_based_tiny_net

from compute_score import compute_score
from utils.CIFAR import CIFAR10Dataset
from utils.subset_classes import subset_classes

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading NAS-Bench-201 API...")
api = API('nb201_api/NAS-Bench-201-v1_1-096897.pth', verbose=False)
print("API loaded.")

dataset = CIFAR10Dataset(data_dir="../data/cifar/cifar-10-batches-py/")
dataset_classes, class_permutation = subset_classes(dataset, samples_per_class=10, device=DEVICE, subsample=100)

all_scores = []
all_accuracies = []

print("Evaluating architectures...")
for index in trange(len(api)):

    config = api.get_net_config(index, 'cifar10')
    network = get_cell_based_tiny_net(config).to(DEVICE)

    info = api.query_meta_info_by_index(index, hp="200")
    accuracy = info.get_metrics("cifar10-valid", 'valid')['accuracy']/100.

    score = compute_score(network, dataset_classes, class_permutation, device=DEVICE)

    all_scores.append(score)
    all_accuracies.append(accuracy)

print("Evaluation completed.")

all_scores = np.array(all_scores)
all_accuracies = np.array(all_accuracies)

plt.scatter(np.log(all_scores), all_accuracies)
plt.xlabel('Log $\lambda_{min}(\Theta)$')
plt.ylabel('Validation Accuracy')
plt.title('Log $\lambda_{min}(\Theta)$ vs Validation Accuracy on NAS-Bench-201 (CIFAR-10)')
plt.tight_layout()
plt.savefig('../../figures/nb201_log_lambda_vs_acc.pdf', bbox_inches='tight', format="pdf")
plt.close()

print("Plot saved as 'nb201_log_lambda_vs_acc.pdf'.")