import torch
import numpy as np

from snn.data_preprocessing.load_data import *


def refractory_period(network):
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.input_neurons), network.alphabet_size], dtype=torch.float).to(network.device),
                torch.zeros([len(network.output_neurons), network.alphabet_size], dtype=torch.float).to(network.device))


def get_acc_and_loss(network, hdf5_group, test_indices, T, classes, input_shape, dt, x_max, polarity):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    outputs = torch.zeros([len(test_indices), network.n_output_neurons, network.alphabet_size, T])

    loss = 0
    true_classes = torch.FloatTensor()

    for j, idx in enumerate(test_indices):
        refractory_period(network)

        inputs, lbl = get_example(hdf5_group, idx, T, classes, input_shape, dt, x_max, polarity)
        true_classes = torch.cat((true_classes, lbl.unsqueeze(0)), dim=0)
        inputs = inputs.to(network.device)

        for t in range(T):
            log_proba = network(inputs[:, :, t].to(network.device))
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[j, :, :, t] = network.spiking_history[network.output_neurons, :, -1].cpu()

    predictions = torch.max(torch.sum(outputs, dim=(-1, -2)), dim=-1).indices
    true_classes = torch.max(torch.sum(true_classes, dim=(-1, -2)), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss
