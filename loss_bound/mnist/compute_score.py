import torch
from torch import nn
from torch.func import vmap, jacrev, functional_call


class Scalar_NN(nn.Module):
    def __init__(self, network, class_val):
        super(Scalar_NN, self).__init__()
        self.network = network
        self.class_val = class_val

    def forward(self, x):
        return self.network(x)[:, self.class_val].reshape(-1, 1)

def get_jacobian(model, x, class_val):

    model = Scalar_NN(network=model, class_val=class_val)
    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0), )).squeeze(0)      

    parameters = {k: v.detach() for k, v in model.named_parameters()}

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False

    jac1 = vmap(jacrev(fnet_single), (None, 0))(parameters, x)
    jac1 = jac1.values()
    jac1 = [j.flatten(1) for j in jac1]

    jac1 = torch.cat(jac1, dim=1)
    return jac1

def compute_score(model, dataset_classes, device="cpu"):
    model = model.to(device)
    jacs = []
    for c in dataset_classes.keys():
        x_ntks = dataset_classes[c].to(device)
        jac = get_jacobian(model, x_ntks, c)
        jacs.append(jac)
    jacs = torch.cat(jacs, dim=0)
    ntk = torch.einsum("Na,Mb->NM", jacs, jacs)
    u, sigma, v = torch.linalg.svd(ntk)
    lambda_min = torch.min(sigma)
    return lambda_min.item(), dataset_classes.keys()



def subset_classes(dataset, samples_per_class=100, device="cuda"):
    dataset_classes = {}
    count_per_class = {}

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

    for key in dataset_classes.keys():
        dataset_classes[key] = torch.stack(dataset_classes[key])

    return dataset_classes