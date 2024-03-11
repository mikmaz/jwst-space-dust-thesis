import torch
import random
import numpy as np
import models.direct
import pickle
from ast import literal_eval


models_dict = {
    'direct_likelihood': models.direct.DirectLikelihoodResNet,
    'direct_univariate_likelihood':
        models.direct.DirectUnivariateLikelihoodResNet,
    'simple_likelihood': models.direct.SimpleDirectLikelihoodNet,
    'fixed_prior': models.direct.FixedPrior
}

filters_to_keep_hmc = [3, 4, 13, 14, 18, 19, 21, 28, 29, 30, 32, 33, 34]


def reconstruct_cov(l_tri_flat, n_features, min_d):
    device = l_tri_flat.device
    l_tri = torch.zeros(
        l_tri_flat.shape[0], n_features, n_features, device=device
    )
    ti = torch.tril_indices(n_features, n_features, 0, device=device)
    l_tri[:, ti[0], ti[1]] = l_tri_flat
    identity = torch.eye(n_features, device=device)
    return torch.bmm(l_tri, l_tri.transpose(1, 2)) + identity * min_d, l_tri


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # TODO change
        return torch.device('cpu')
    else:
        return torch.device('cpu')


def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def none_literal_eval(dict_str):
    return literal_eval(dict_str) if dict_str is not None else None
