import torch
import numpy as np
from data_preprocessing.load_data import *


def refractory_period(network):
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.visible_neurons), network.alphabet_size], dtype=torch.float).to(network.device))


def get_acc_and_loss(network, hdf5_group, test_indices, T, n_classes, input_shape, dt, x_max, polarity):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    outputs = torch.zeros([len(test_indices), network.n_output_neurons, network.alphabet_size, T])

    loss = 0
    true_classes = hdf5_group.labels[test_indices, 0]

    for j, idx in enumerate(test_indices):
        refractory_period(network)

        inputs, lbl = get_example(hdf5_group, idx, T, n_classes, input_shape, dt, x_max, polarity)

        for t in range(T):
            log_proba = network(inputs[:, :, t].to(network.device))
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[j, :, :, t] = network.spiking_history[network.output_neurons, :, -1]

    predictions = torch.max(torch.sum(outputs, dim=(-1, -2)), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss
