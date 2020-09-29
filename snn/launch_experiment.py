from __future__ import print_function
import pickle

import numpy as np
import tables
import argparse
import torch

from snn.utils.misc import *
from snn.experiments import binary_exp, wta_exp

''''
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--dataset', default='mnist_dvs')
    parser.add_argument('--model', default='snn', choices=['snn', 'wta', 'wispike'], help='Model type, either "binary" or "wta"')
    parser.add_argument('--num_ite', default=5, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--test_period', default=1000, type=int, help='')

    parser.add_argument('--dt', default=25000, type=int, help='')
    parser.add_argument('--sample_length', default=2000, type=int, help='')
    parser.add_argument('--input_shape', nargs='+', default=[1352], type=int, help='Shape of an input sample')
    parser.add_argument('--polarity', default='true', type=str, help='Use polarity or not')

    parser.add_argument('--num_samples_train', default=1, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=1, type=int, help='Number of samples to test on')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--start_idx', type=int, default=0, help='When resuming training from existing weights, index to start over from')
    parser.add_argument('--labels', nargs='+', default=None, type=int, help='Class labels to be used during training')
    parser.add_argument('--pattern', nargs='+', default=None, type=int, help='Class labels to be used during training')

    parser.add_argument('--home', default='/home')
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--weights', type=str, default=None, help='Path to existing weights (relative to home)')

    parser.add_argument('--record_test_acc', type=str, default='true', help='')
    parser.add_argument('--record_test_loss', type=str, default='false', help='')
    parser.add_argument('--record_train_loss', type=str, default='false', help='')
    parser.add_argument('--record_train_acc', type=str, default='false', help='')
    parser.add_argument('--record_all', type=str, default='false', help='Overrides the other record arguments to set them to yes')
    parser.add_argument('--suffix', type=str, default='', help='Appended to the name of the saved results and weights')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')


    # Arguments common to all models
    parser.add_argument('--n_h', default=256, type=int, help='Number of hidden neurons')
    parser.add_argument('--topology_type', default='fully_connected', type=str, choices=['fully_connected', 'feedforward', 'layered', 'custom'], help='Topology of the network')
    parser.add_argument('--density', default=None, type=float, help='Density of the connections if topology_type is "sparse"')
    parser.add_argument('--n_neurons_per_layer', default=0, type=int, help='Number of neurons per layer if topology_type is "layered"')
    parser.add_argument('--initialization', default='uniform', type=str, choices=['uniform', 'glorot'], help='Initialization of the weights')
    parser.add_argument('--weights_magnitude', default=0.05, type=float, help='Magnitude of weights at initialization')

    parser.add_argument('--n_basis_ff', default=8, type=int, help='Number of basis functions for synaptic connections')
    parser.add_argument('--syn_filter', default='raised_cosine_pillow_08', type=str,
                        choices=['base_filter', 'cosine_basis', 'raised_cosine', 'raised_cosine_pillow_05', 'raised_cosine_pillow_08'],
                        help='Basis function to use for synaptic connections')
    parser.add_argument('--tau_ff', default=10, type=int, help='Feedforward connections time constant')
    parser.add_argument('--n_basis_fb', default=1, type=int, help='Number of basis functions for feedback connections')
    parser.add_argument('--tau_fb', default=10, type=int, help='Feedback connections time constant')
    parser.add_argument('--mu', default=1.5, type=float, help='Width of basis functions')

    parser.add_argument('--kappa', default=0.2, type=float, help='eligibility trace decay coefficient')
    parser.add_argument('--r', default=0.3, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization strength')

    args = parser.parse_args()

print(args)

datasets = {'mnist_dvs': r'mnist_dvs_events.hdf5',
            'dvs_gesture': r'dvs_gestures_events.hdf5'
            }

if args.dataset[:5] == 'mnist':
    dataset = args.home + r'/datasets/mnist-dvs/' + datasets[args.dataset]
elif args.dataset[:11] == 'dvs_gesture':
    dataset = args.home + r'/datasets/DvsGesture/' + datasets[args.dataset]
else:
    print('Error: dataset not found')

dataset = tables.open_file(dataset)


### Learning parameters
if not args.num_samples_train:
    args.num_samples_train = dataset.root.stats.train_data[0]

# Save results and weights
name = args.dataset + r'_' + args.model + r'_%d_epochs_nh_%d_dt_%d_' % (args.num_samples_train, args.n_h, args.dt) + r'_pol_' + args.polarity + args.suffix

results_path = args.home + r'/results/'

if args.weights is None:
    if args.save_path is None:
        args.save_path = mksavedir(pre=results_path, exp_dir=name)
else:
    args.save_path = args.weights

with open(args.save_path + 'commandline_args.pkl', 'wb') as f:
    pickle.dump(args.__dict__, f, pickle.HIGHEST_PROTOCOL)

args.dataset = dataset

# Select training and test examples from subset of labels if specified
get_indices(args)
make_recordings(args)

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.polarity = str2bool(args.polarity)
args.n_classes = args.dataset.root.stats.test_label[1]

### Network parameters
if args.polarity:
    args.n_input_neurons = int(2 * (args.dataset.root.stats.train_data[1] ** 2))
else:
    args.n_input_neurons = int(args.dataset.root.stats.train_data[1] ** 2)
args.n_output_neurons = args.dataset.root.stats.train_label[1]
args.n_hidden_neurons = args.n_h


if args.topology_type == 'custom':
    args.topology = torch.zeros([args.n_hidden_neurons + args.n_output_neurons,
                                 args.n_input_neurons + args.n_hidden_neurons + args.n_output_neurons])
    args.topology[-args.n_output_neurons:, args.n_input_neurons:-args.n_output_neurons] = 1
    args.topology[:args.n_hidden_neurons, :(args.n_input_neurons + args.n_hidden_neurons)] = 1
    # Feel free to fill this with any custom topology
    print(args.topology)
else:
    args.topology = None

# Create the network
if args.model == 'snn':
    binary_exp.launch_binary_exp(args)
elif args.model == 'wta':
    wta_exp.launch_multivalued_exp(args)
else:
    raise NotImplementedError('Please choose a model between "snn" and "wta"')

