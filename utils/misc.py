import torch
import numpy as np
import argparse
import time
import os
import fnmatch
from utils import filters
import pickle


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


def time_average(old, new, kappa):
    return old * kappa + new * (1 - kappa)


def find_indices_for_labels(hdf5_group, labels):
    res = []
    for label in labels:
        res.append(np.where(hdf5_group.labels[:, 0] == label)[0])
    return np.hstack(res)


def get_indices(args):
    if args.labels is not None:
        print('Training on labels ', args.labels)
        indices = np.random.choice(find_indices_for_labels(args.dataset.root.train, args.labels), [args.num_samples_train], replace=True)
        args.num_samples_test = min(args.num_samples_test, len(find_indices_for_labels(args.dataset.root.test, args.labels)))
        test_indices = np.random.choice(find_indices_for_labels(args.dataset.root.test, args.labels), [args.num_samples_test], replace=False)
    else:
        indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True)
        test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

    return indices, test_indices


def make_topology(network_type, topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, n_neurons_per_layer=0, topology=None, density=1):
    if topology_type == 'fully_connected':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons], dtype=torch.float)
        assert torch.sum(topology[:, :n_input_neurons]) == (n_input_neurons * (n_hidden_neurons + n_output_neurons))

    if topology_type == 'feedforward':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[-n_output_neurons:, :n_input_neurons] = 0
        topology[:n_hidden_neurons, -n_output_neurons:] = 0

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

        topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    elif topology_type == 'custom':
        topology = topology


    if (network_type == 'snn') & (topology_type != 'feedforward') & (topology_type != 'custom'):
        topology[:n_hidden_neurons, -n_output_neurons:] = 1
    if network_type == 'wta':
        topology[:n_hidden_neurons, -n_output_neurons:] = 0
    topology[-n_output_neurons:, -n_output_neurons:] = 1

    # Make sure that the diagonal is all 0s
    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    return topology


def make_network_parameters(network_type, n_input_neurons, n_output_neurons, n_hidden_neurons, topology_type='fully_connected', topology=None, n_neurons_per_layer=0,
                            density=1, weights_magnitude=0.05, initialization='uniform', synaptic_filter=filters.raised_cosine_pillow_08, n_basis_ff=8,
                            n_basis_fb=1, tau_ff=10, tau_fb=10, mu=1.5):

    topology = make_topology(network_type, topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, n_neurons_per_layer, topology, density)
    print(topology[:, n_input_neurons:])

    network_parameters = {'n_input_neurons': n_input_neurons,
                          'n_output_neurons': n_output_neurons,
                          'n_hidden_neurons': n_hidden_neurons,
                          'topology': topology,
                          'synaptic_filter': synaptic_filter,
                          'n_basis_feedforward': n_basis_ff,
                          'n_basis_feedback': n_basis_fb,
                          'tau_ff': tau_ff,
                          'tau_fb': tau_fb,
                          'mu': mu,
                          'initialization': initialization,
                          'weights_magnitude': weights_magnitude,
                          }

    return network_parameters


def mksavedir(pre='results/', exp_dir=None):
    """
    Creates a results directory in the subdirectory 'pre'
    """

    if pre[-1] != '/':
        pre + '/'

    if not os.path.exists(pre):
        os.makedirs(pre)
    prelist = np.sort(fnmatch.filter(os.listdir(pre), '[0-9][0-9][0-9]__*'))

    if exp_dir is None:
        if len(prelist) == 0:
            expDirN = "001"
        else:
            expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

        save_dir = time.strftime(pre + expDirN + "__" + "%d-%m-%Y", time.localtime())

    elif isinstance(exp_dir, str):
        if len(prelist) == 0:
            expDirN = "001"
        else:
            expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

        save_dir = time.strftime(pre + expDirN + "__" + "%d-%m-%Y", time.localtime()) + '_' + exp_dir

    else:
        raise TypeError('exp_dir should be a string')


    os.makedirs(save_dir)
    print(("Created experiment directory {0}".format(save_dir)))
    return save_dir + r'/'


def save_results(results, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

