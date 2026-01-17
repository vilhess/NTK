import csv
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd

import sys 
sys.path.append('../')

from nats_bench import create
from xautodl.models import get_cell_based_tiny_net

from compute_score import compute_score
from utils.CIFAR import CIFAR10Dataset
from utils.subset_classes import subset_classes
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Architecture Search evaluation')
    parser.add_argument('--mode', type=str, default='tss', choices=['tss', 'sss'],
                        help='NATS-Bench mode: "tss" or "sss" (default: tss)')
    parser.add_argument('--continue_from', action='store_true', default=False,
                        help='Continue from the last index in the CSV file (default: False)')

    args = parser.parse_args()
    mode = args.mode
    mode_description = "Topology" if mode == 'tss' else "Size"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading NAS-Bench-201 API in {mode_description} mode...")
    modes = {'tss': 'NATS-tss-v1_0-3ffb9-simple',
            'sss': 'NATS-sss-v1_0-50262-simple',}
    hps = {'tss': '200', 'sss': '90'}

    api = create(f"nats/{modes[mode]}", mode, fast_mode=True, verbose=False)
    print("API loaded.")

    dataset = CIFAR10Dataset(data_dir="../data/cifar/cifar-10-batches-py/")
    dataset_classes, class_permutation = subset_classes(dataset, samples_per_class=10, device=DEVICE, subsample=100)

    all_scores = []
    all_accuracies = []

    csv_path = f"../results/nas/nas_{mode}_history.csv"
    
    # Determine starting index based on --continue_from flag
    start_index = 0
    if args.continue_from and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            start_index = int(df["index"].max()) + 1
            print(f"Continuing from index {start_index}...")
    else:
        # If not continuing, remove the file to start fresh
        if os.path.exists(csv_path):
            os.remove(csv_path)


    print("Evaluating architectures...")
    for index in trange(start_index, len(api)):

        config = api.get_net_config(index, 'cifar10')
        network = get_cell_based_tiny_net(config).to(DEVICE)

        info = api.query_meta_info_by_index(index, hp=hps[mode])
        accuracy = info.get_metrics("cifar10-valid", 'valid')['accuracy']/100.

        score = compute_score(network, dataset_classes, class_permutation, device=DEVICE)


        row = {
            "index": index,
            "accuracy": float(accuracy),
            "score": float(score),
        }
        # Store as JSON strings in CSV cells
        row = {k: json.dumps(v) for k, v in row.items()}

        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "accuracy", "score"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        del network
        del score
        torch.cuda.empty_cache()

    print("Evaluation completed.")