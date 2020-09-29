from __future__ import print_function
import pickle

import numpy as np
import tables
import argparse
import torch
import yaml

from snn.utils.misc import *
from snn.experiments import binary_exp, wta_exp

''''
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default='/home')
    parser.add_argument('--params_file', default='/snn/snn/parameters/params_mnistdvs_binary.yml')
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--weights', type=str, default=None, help='Path to existing weights (relative to home)')

    # Arguments common to all models

    args = parser.parse_args()

print(args)

# Save results and weights
results_path = args.home + r'/results/'

if args.weights is None:
    if args.save_path is None:
        params_file = args.home + args.params_file
        with open(params_file, 'r') as f:
            params = yaml.load(f)

        name = params['dataset'].split("/", -1)[-1][:-5] + r'_' + params['model'] \
               + r'_%d_epochs_nh_%d_dt_%d_' % (params['num_samples_train'], params['n_h'], params['dt']) + r'_pol_' + str(params['polarity']) + params['suffix']

        args.save_path = mksavedir(pre=results_path, exp_dir=name)

        with open(args.save_path + '/params.yml', 'w') as outfile:
            yaml.dump(params_file, outfile, default_flow_style=False)
else:
    args.save_path = args.home + args.weights
    params_file = args.save_path + '/params.yml'

    with open(params_file, 'r') as f:
        params = yaml.load(f)

dataset = tables.open_file(args.home + params['dataset'])


### Learning parameters
if params['num_samples_train'] is None:
    params['num_samples_train'] = dataset.root.stats.train_data[0]

params['dataset'] = dataset

# Select training and test examples from subset of labels if specified
get_indices(params)
make_recordings(args, params)

args.device = None
if not params['disable_cuda'] and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

params['n_classes'] = params['dataset'].root.stats.test_label[1]

### Network parameters
if params['polarity']:
    params['n_input_neurons'] = int(2 * (params['dataset'].root.stats.train_data[1] ** 2))
else:
    params['n_input_neurons'] = int(params['dataset'].root.stats.train_data[1] ** 2)
params['n_output_neurons'] = params['dataset'].root.stats.train_label[1]
params['n_hidden_neurons'] = params['n_h']


if params['topology_type'] == 'custom':
    params['topology'] = torch.zeros([args.n_hidden_neurons + args.n_output_neurons,
                                 args.n_input_neurons + args.n_hidden_neurons + args.n_output_neurons])
    params['topology'][-args.n_output_neurons:, args.n_input_neurons:-args.n_output_neurons] = 1
    params['topology'][:args.n_hidden_neurons, :(args.n_input_neurons + args.n_hidden_neurons)] = 1
    # Feel free to fill this with any custom topology
    print(params['topology'])
else:
    params['topology'] = None

# Create the network
if params['model'] == 'snn':
    binary_exp.launch_binary_exp(args, params)
elif params['model'] == 'wta':
    wta_exp.launch_multivalued_exp(args, params)
else:
    raise NotImplementedError('Please choose a model between "snn" and "wta"')

