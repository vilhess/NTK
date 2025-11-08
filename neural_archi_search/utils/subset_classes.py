import torch
from torch.utils.data import Dataset
import random

# Return a dictionnary : key -> class / item -> torch batch of samples from this class
def subset_classes(dataset: Dataset, samples_per_class=10, device="cpu", subsample=12):
    dataset_classes = {}
    count_per_class = {}
    class_permutation = None

    for inp, tar in dataset:
        try:
            if tar not in dataset_classes:
                dataset_classes[tar] = []
                count_per_class[tar] = 0
            if count_per_class[tar] < samples_per_class:
                dataset_classes[tar].append(inp.to(device))
                count_per_class[tar] += 1
        except    Exception as e:
            print(f"Error with target {tar} : {e}")

        if all(count >= samples_per_class for count in count_per_class.values()):
            break

    if len(dataset_classes) > subsample:
        selected_classes = random.sample(list(dataset_classes.keys()), subsample)
        dataset_classes = {key: dataset_classes[key] for key in selected_classes}
        class_permutation = {selected_classes[i]: i for i in range(len(selected_classes))}

    for key in dataset_classes.keys():
        dataset_classes[key] = torch.stack(dataset_classes[key])

    return dataset_classes, class_permutation
