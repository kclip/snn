import torch
import numpy as np
import utils.filters as filters
import tables
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def custom_softmax(input_tensor, alpha, dim_):
    u = torch.max(input_tensor)
    return torch.exp(alpha * (input_tensor - u)) / (torch.exp(- alpha * u) + torch.sum(torch.exp(alpha * (input_tensor - u)), dim=dim_))[:, None]


def make_topology(mode, topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, n_neurons_per_layer=0, density=1):
    if topology_type == 'fully_connected':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons], dtype=torch.float)

    elif topology_type == 'sparse':
        indices = np.random.choice(n_hidden_neurons * n_hidden_neurons, [int(density * n_hidden_neurons**2)], replace=False)

        row = np.array([int(index / n_hidden_neurons) for index in indices])
        col = np.array([int(index % n_hidden_neurons) for index in indices]) + n_input_neurons

        topology = torch.zeros([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[[r for r in row], [c for c in col]] = 1
        topology[:, :n_input_neurons] = 1
        topology[-n_output_neurons:, :] = 1

    elif topology_type == 'layered':
        n_layers = n_hidden_neurons // n_neurons_per_layer
        assert (n_hidden_neurons % n_neurons_per_layer) == 0

        topology = torch.zeros([(n_hidden_neurons + n_output_neurons), (n_input_neurons + n_hidden_neurons + n_output_neurons)])

        topology[:, :n_input_neurons] = 1

        for i in range(1, n_layers + 1):
            topology[i * n_neurons_per_layer: (i + 1) * n_neurons_per_layer, n_input_neurons + (i - 1) * n_neurons_per_layer: n_input_neurons + i * n_neurons_per_layer] = 1

        topology[:n_hidden_neurons, -n_output_neurons:] = 0
        topology[-n_output_neurons:, -n_output_neurons:] = 1
        topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    if mode == 'train_ml':
        topology[:n_hidden_neurons, -n_output_neurons:] = 0
    elif mode == 'train_rl':
        topology[:n_hidden_neurons, -n_output_neurons:] = 1

    topology[-n_output_neurons:, -n_output_neurons:] = 1
    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    assert torch.sum(topology[:, :n_input_neurons]) == (n_hidden_neurons + n_output_neurons) * n_input_neurons
    return topology


def make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode, topology_type='fully_connected', topology=None, n_neurons_per_layer=0,
                            density=1, weights_magnitude=0.05, initialization='uniform', connection_topology='full', n_basis_ff=8, ff_filter=filters.raised_cosine_pillow_08,
                            n_basis_fb=1, fb_filter=filters.raised_cosine_pillow_08, tau_ff=10, tau_fb=10, mu=1.5, dropout_rate=None):

    if topology_type != 'custom':
        topology = make_topology(mode, topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, n_neurons_per_layer, density)
    else:
        topology = topology

    print(topology[:, n_input_neurons:])
    network_parameters = {'n_input_neurons': n_input_neurons,
                          'n_output_neurons': n_output_neurons,
                          'n_hidden_neurons': n_hidden_neurons,
                          'topology': topology,
                          'alphabet_size': alphabet_size,
                          'n_basis_feedforward': n_basis_ff,
                          'feedforward_filter': ff_filter,
                          'n_basis_feedback': n_basis_fb,
                          'feedback_filter': fb_filter,
                          'tau_ff': tau_ff,
                          'tau_fb': tau_fb,
                          'mu': mu,
                          'initialization': initialization,
                          'connection_topology': connection_topology,
                          'weights_magnitude': weights_magnitude,
                          'mode': mode,
                          'dropout_rate': dropout_rate
                          }

    return network_parameters


def refractory_period(network):
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.visible_neurons), network.alphabet_size], dtype=torch.float).to(network.device))


def time_average(old, new, kappa):
    return old * kappa + new * (1 - kappa)


def get_acc_and_loss(network, dataset, test_indices):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.set_mode('test')
    network.reset_internal_state()

    S_prime = dataset.root.stats.test[:][-1]
    outputs = torch.zeros([len(test_indices), network.n_output_neurons, network.alphabet_size, S_prime])

    loss = 0

    for j, sample_idx in enumerate(test_indices):
        refractory_period(network)

        sample = torch.FloatTensor(dataset.root.test.data[sample_idx])

        for s in range(S_prime):
            log_proba = network(sample[:, :, s].to(network.device))

            loss += torch.sum(log_proba).numpy()
            outputs[j, :, :, s % S_prime] = network.spiking_history[network.output_neurons, :, -1]

    predictions = torch.max(torch.sum(outputs, dim=(-1, -2)), dim=-1).indices
    true_classes = torch.max(torch.sum(torch.FloatTensor(dataset.root.test.label[:][test_indices]), dim=(-1, -2)), dim=-1).indices

    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss


def get_train_acc_and_loss(network, dataset):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.set_mode('test')
    network.reset_internal_state()

    train_shape = dataset.root.stats.train[:]
    S_prime = train_shape[-1]
    indices = [i for i in range(train_shape[0])]
    outputs = torch.zeros([len(indices), network.n_output_neurons, network.alphabet_size, S_prime])

    loss = 0

    for j, sample_idx in enumerate(indices):
        refractory_period(network)

        sample = torch.FloatTensor(dataset.root.train.data[sample_idx])

        for s in range(S_prime):
            log_proba = network(sample[:, :, s].to(network.device))
            loss += torch.sum(log_proba).numpy()
            outputs[j, :, :, s % S_prime] = network.spiking_history[network.output_neurons, :, -1]

    predictions = torch.max(torch.sum(outputs, dim=(-1, -2)), dim=-1).indices
    true_classes = torch.max(torch.sum(torch.FloatTensor(dataset.root.train.label[:][indices]), dim=(-1, -2)), dim=-1).indices

    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss


def find_train_indices_for_labels(dataset, labels):
    res = []
    for label in labels:
        res.append(np.where(np.argmax(np.sum(tables.open_file(dataset).root.train.label[:], axis=(-1, -2)), axis=-1) == label)[0])
    return np.hstack(res)


def find_test_indices_for_labels(dataset, labels):
    res = []
    for label in labels:
        res.append(np.where(np.argmax(np.sum(tables.open_file(dataset).root.test.label[:], axis=(-1, -2)), axis=-1) == label)[0])
    return np.hstack(res)
