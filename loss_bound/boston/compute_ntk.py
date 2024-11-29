import torch
from torch.func import functional_call, vmap, jacrev

def compute_ntk(model, x):
    parameters = {k: v.detach() for k, v in model.named_parameters()}

    def fnet_single(params, x):
        return functional_call(model, params, x)

    jac1 = vmap(jacrev(fnet_single), (None, 0))(parameters, x)
    jac1 = tuple(jac1[name] for name in jac1)
    jac1 = [j.flatten(1) for j in jac1]
    result = torch.stack([torch.einsum('Na, Ma ->NM', j, j) for j in jac1]).sum(0)
    return result