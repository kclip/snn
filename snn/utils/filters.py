import torch
import numpy as np


# Simple exponentially decreasing filter
def base_filter(t, n_basis, tau_1):
    return torch.tensor(np.vstack([np.exp([-i / tau_1 for i in range(t)]) for _ in range(n_basis)]), dtype=torch.float)


# Simple cosine basis
def cosine_basis(T, n_basis, mu):
    # mu: compression factor
    c_intervals = T / max(n_basis - 1, 1)
    c = np.arange(0, T + c_intervals, c_intervals)

    return torch.FloatTensor([[0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * mu * (t - c[k]) / c_intervals))) / 2 for t in range(T)] for k in range(n_basis)])


# Raised cosine as coded by Hyeryung
def raised_cosine(T, n_basis, mu):
    # mu = dilation factor
    b = np.array([k * np.pi / 2 + np.pi for k in range(n_basis)])
    tau = T
    a = (b + np.pi) / np.log(tau + mu)

    return torch.FloatTensor([[0.5 * np.cos(max(-np.pi, min(np.pi, a[k] * np.log(t + mu) - b[k]))) + 0.5 for t in range(T)] for k in range(n_basis)])


# Raised cosine as defined in Pillow 2005
def raised_cosine_pillow_05(T, n_basis, mu):
    # mu = dilation factor, when mu = T, all the cosines have the same width
    c_min = np.log(mu)
    c_max = np.log(T + mu)
    c_intervals = (c_max - c_min) / max(n_basis - 1, 1)
    c = np.arange(c_min, c_max + c_intervals, c_intervals)
    b = 1
    return torch.FloatTensor([[0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (np.log(t + b) - c[k]) / c_intervals))) / 2 for t in range(T)] for k in range(n_basis)])


# Raised cosine as defined in Pillow 2008
def raised_cosine_pillow_08(T, n_basis, mu):
    # mu = compression parameter
    phi = np.array([k * np.pi / 2 for k in range(n_basis)])
    b = 1

    return torch.FloatTensor([[0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * mu * np.log(t + b) - phi[k]))) / 2 for t in range(T)] for k in range(n_basis)])


def get_filter(selected_filter):
    filters_dict = {'base_filter': base_filter, 'cosine_basis': cosine_basis,
                    'raised_cosine': raised_cosine, 'raised_cosine_pillow_05': raised_cosine_pillow_05, 'raised_cosine_pillow_08': raised_cosine_pillow_08}

    return filters_dict[selected_filter]

