import torch

def get_relative_norm(ntk_t, ntk_init):
    rel_norm = torch.linalg.norm(ntk_t - ntk_init)**2 / torch.linalg.norm(ntk_init)**2
    return rel_norm.item()