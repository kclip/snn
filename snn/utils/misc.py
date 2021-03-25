import pickle
import time
import os

import torch
import numpy as np
import argparse
import fnmatch

from . import filters


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
        # Make sure that the diagonal is all 0s
        topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0
        return topology


    if (network_type == 'snn') & (topology_type != 'feedforward'):
        topology[:n_hidden_neurons, -n_output_neurons:] = 1
        topology[-n_output_neurons:, -n_output_neurons:] = 1

    if network_type == 'wta':
        topology[:n_hidden_neurons, -n_output_neurons:] = 0
        topology[-n_output_neurons:, -n_output_neurons:] = 1

    # Make sure that the diagonal is all 0s
    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    return topology


def make_network_parameters(network_type, n_input_neurons, n_output_neurons, n_hidden_neurons, topology_type='fully_connected', topology=None, n_neurons_per_layer=0,
                            density=1, weights_magnitude=0.05, initialization='uniform', synaptic_filter=filters.raised_cosine_pillow_08, n_basis_ff=8,
                            n_basis_fb=1, tau_ff=10, tau_fb=10, mu=0.5):

    topology = make_topology(network_type, topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, n_neurons_per_layer, topology, density)
    print(topology)

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


def make_recordings(args, params):
    params['record_test_acc'] = params['record_test_acc']
    params['record_test_loss'] = params['record_test_loss']
    params['record_train_loss'] = params['record_train_loss']
    params['record_train_acc'] = params['record_train_acc']

    if params['record_all']:
        params['record_test_acc'] = True
        params['record_test_loss'] = True
        params['record_train_loss'] = True
        params['record_train_acc'] = True

    if params['test_period'] is not None:
        params['ite_test'] = np.arange(0, params['n_examples_train'], params['test_period'])

        if args.weights is not None:
            if params['record_test_acc']:
                with open(args.home + args.weights + '/test_accs.pkl', 'rb') as f:
                    test_accs = pickle.load(f)
            else:
                test_accs = None
            if params['record_test_loss']:
                with open(args.home + args.weights + '/test_losses.pkl', 'rb') as f:
                    test_losses = pickle.load(f)
            else:
                test_losses = None
            if params['record_train_loss']:
                with open(args.home + args.weights + '/train_losses.pkl', 'rb') as f:
                    train_losses = pickle.load(f)
            else:
                train_losses = None
            if params['record_train_acc']:
                with open(args.home + args.weights + '/train_accs.pkl', 'rb') as f:
                    train_accs = pickle.load(f)
            else:
                train_accs = None
        else:
            if params['record_test_acc']:
                test_accs = {i: [] for i in params['ite_test']}
                test_accs[params['n_examples_train']] = []
            else:
                test_accs = None
            if params['record_test_loss']:
                test_losses = {i: [] for i in params['ite_test']}
                test_losses[params['n_examples_train']] = []
            else:
                test_losses = None
            if params['record_train_loss']:
                train_losses = {i: [] for i in params['ite_test']}
                train_losses[params['n_examples_train']] = []
            else:
                train_losses = None
            if params['record_train_acc']:
                train_accs = {i: [] for i in params['ite_test']}
                train_accs[params['n_examples_train']] = []
            else:
                train_accs = None
    return train_accs, train_losses, test_accs, test_losses


def save_results(results, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

