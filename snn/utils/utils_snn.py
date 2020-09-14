import torch
import numpy as np

from snn.data_preprocessing.load_data import *


def refractory_period(network):
    """"
    Neural refractory period between two samples
    """
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.visible_neurons)], dtype=torch.float).to(network.device))


def get_acc_and_loss(network, hdf5_group, test_indices, T, n_classes, input_shape, dt, x_max, polarity):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])
    loss = 0

    true_classes = torch.LongTensor(hdf5_group.labels[test_indices, 0])

    for j, idx in enumerate(test_indices):
        refractory_period(network)

        inputs, lbl = get_example(hdf5_group, idx, T, n_classes, input_shape, dt, x_max, polarity)
        inputs = inputs.to(network.device)

        for t in range(T):
            log_proba = network(inputs[:, t])
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[j, :, t] = network.spiking_history[network.output_neurons, -1].cpu()

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    print(predictions[:10])
    print(true_classes[:10])

    return acc, loss
