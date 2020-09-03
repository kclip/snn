import torch
import numpy as np
import utils.filters as filters
import pickle
from data_preprocessing.load_data import *


def find_indices_for_labels(hdf5_group, labels):
    res = []
    for label in labels:
        res.append(np.where(np.argmax(np.sum(hdf5_group.label[:], axis=-1), axis=-1) == label)[0])
    return np.hstack(res)


def make_topology(topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, topology=None, density=1):
    if topology_type == 'feedforward':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[-n_output_neurons:, :n_input_neurons] = 0
        topology[:n_hidden_neurons, -n_output_neurons:] = 0

    if topology_type == 'no_backward':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[n_hidden_neurons:, -n_output_neurons:] = 0

    elif topology_type == 'fully_connected':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons], dtype=torch.float)
        assert torch.sum(topology[:, :n_input_neurons]) == (n_input_neurons * (n_hidden_neurons + n_output_neurons))

    elif topology_type == 'sparse':
        indices = np.random.choice(n_hidden_neurons * n_hidden_neurons, [int(density * n_hidden_neurons**2)], replace=False)

        row = np.array([int(index / n_hidden_neurons) for index in indices])
        col = np.array([int(index % n_hidden_neurons) for index in indices]) + n_input_neurons

        topology = torch.zeros([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[[r for r in row], [c for c in col]] = 1
        topology[:, :n_input_neurons] = 1
        topology[-n_output_neurons:, :] = 1

    elif topology_type == 'custom':
        topology = topology

    if (topology_type != 'feedforward') & (topology_type != 'custom'):
        topology[:n_hidden_neurons, -n_output_neurons:] = 1
        topology[-n_output_neurons:, -n_output_neurons:] = 1

    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0


    return topology


def make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, topology_type='fully_connected', topology=None, density=1, mode='train', weights_magnitude=0.05,
                            n_basis_ff=8, ff_filter=filters.raised_cosine_pillow_08, n_basis_fb=1, fb_filter=filters.raised_cosine_pillow_08, initialization='uniform',
                            tau_ff=10, tau_fb=10, mu=1.5, save_path=None):

    topology = make_topology(topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, topology, density)
    print(topology[:, n_input_neurons:])

    network_parameters = {'n_input_neurons': n_input_neurons,
                          'n_output_neurons': n_output_neurons,
                          'n_hidden_neurons': n_hidden_neurons,
                          'topology': topology,
                          'n_basis_feedforward': n_basis_ff,
                          'feedforward_filter': ff_filter,
                          'n_basis_feedback': n_basis_fb,
                          'feedback_filter': fb_filter,
                          'tau_ff': tau_ff,
                          'tau_fb': tau_fb,
                          'mu': mu,
                          'initialization': initialization,
                          'weights_magnitude': weights_magnitude,
                          'mode': mode,
                          'save_path': save_path
                          }

    return network_parameters


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
    network.set_mode('test')
    network.reset_internal_state()

    outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])
    loss = 0

    rec = torch.zeros([network.n_learnable_neurons, T])
    labels = torch.zeros([len(test_indices), n_classes, T])

    for j, idx in enumerate(test_indices):
        refractory_period(network)

        inputs, lbl = get_example(hdf5_group, idx, T, n_classes, input_shape, dt, x_max, polarity)
        labels[j] = lbl

        for t in range(T):
            log_proba = network(inputs[:, t])
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[j, :, t] = network.spiking_history[network.output_neurons, -1]
            rec[:, t] = network.spiking_history[network.learnable_neurons, -1]

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(labels, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss


def save_results(results, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
