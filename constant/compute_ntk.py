import torch
from torch.func import vmap, jacrev, functional_call


def get_ntk(fnet_single, params, x_ntk, multi=True):
    jac = vmap(jacrev(fnet_single), (None, 0))(params, x_ntk)
    jac = jac.values()

    if multi: # Classification
        jac = [j.flatten(2) for j in jac]
        result = torch.stack([torch.einsum("Naf, Mbf -> NMab", j, j) for j in jac])

    else: # Regression
        jac = [j.flatten(1) for j in jac]
        result = torch.stack([torch.einsum("Na, Mb -> NM", j, j) for j in jac])

    result = result.sum(0)

    return result


def get_fnet_single(model):

    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0), )).squeeze(0)
    
    return fnet_single
    