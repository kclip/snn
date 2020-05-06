from __future__ import print_function
import torch
from multivalued_snn.utils_multivalued.misc import str2bool
from multivalued_snn import multivalued_exp
from binary_snn import binary_exp
import time
import numpy as np
import tables
import pickle
import argparse
import os

''''
Train a WTA-SNN with VOWEL.
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--model', default='train_ml_online', choices=['binary', 'wta'], help='Model type, either "binary" or "wta"')
    parser.add_argument('--num_ite', default=5, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_train', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--start_idx', type=int, default=0, help='When resuming training from existing weights, index to start over from')
    parser.add_argument('--suffix', type=str, default='', help='Appended to the name of the saved results and weights')
    parser.add_argument('--labels', nargs='+', default=None, type=int, help='Class labels to be used during training')


    # Arguments common to all models
    parser.add_argument('--n_h', default=128, type=int, help='Number of hidden neurons')
    parser.add_argument('--topology_type', default='fully_connected', type=str, choices=['fully_connected', 'feedforward', 'layered', 'custom'], help='Topology of the network')
    parser.add_argument('--density', default=None, type=int, help='Density of the connections if topology_type is "sparse"')
    parser.add_argument('--initialization', default='uniform', type=str, choices=['uniform', 'glorot'], help='Initialization of the weights')
    parser.add_argument('--weights_magnitude', default=0.05, type=float, help='Magnitude of weights at initialization')

    parser.add_argument('--n_basis_ff', default=8, type=int, help='Number of basis functions for synaptic connections')
    parser.add_argument('--ff_filter', default='raised_cosine_pillow_08', type=str,
                        choices=['base_ff_filter', 'base_fb_filter', 'cosine_basis', 'raised_cosine', 'raised_cosine_pillow_05', 'raised_cosine_pillow_08'],
                        help='Basis function to use for synaptic connections')
    parser.add_argument('--tau_ff', default=10, type=int, help='Feedforward connections time constant')
    parser.add_argument('--n_basis_fb', default=1, type=int, help='Number of basis functions for feedback connections')
    parser.add_argument('--fb_filter', default='raised_cosine_pillow_08', type=str,
                        choices=['base_ff_filter', 'base_fb_filter', 'cosine_basis', 'raised_cosine', 'raised_cosine_pillow_05', 'raised_cosine_pillow_08'],
                        help='Basis function to use for feedback connections')
    parser.add_argument('--tau_fb', default=10, type=int, help='Feedback connections time constant')
    parser.add_argument('--mu', default=1.5, type=float, help='Width of basis functions')

    parser.add_argument('--kappa', default=0.2, type=float, help='eligibility trace decay coefficient')
    parser.add_argument('--r', default=0.8, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization strength')


    # Arguments for WTA models
    parser.add_argument('--dropout_rate', default=None, type=float, help='')
    parser.add_argument('--T', type=float, default=1., help='temperature')
    parser.add_argument('--n_neurons_per_layer', default=None, type=int, help='Number of neurons per layer if topology_type is "layered"')

    args = parser.parse_args()

print(args)

local_data_path = r'/path/to/datasets'

if args.dataset[:3] == 'shd':
    dataset = local_data_path + r'/shd/' + args.dataset
elif args.dataset[:5] == 'mnist':
    dataset = local_data_path + r'/mnist-dvs/' + args.dataset
elif args.dataset[:11] == 'dvs_gesture':
    dataset = local_data_path + r'/DvsGesture/' + args.dataset
elif args.dataset[:7] == 'swedish':
    dataset = local_data_path + r'/SwedishLeaf_processed/' + args.dataset
else:
    print('Error: dataset not found')

args.dataset = tables.open_file(dataset)


args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


### Network parameters
args.n_input_neurons = dataset.root.stats.train_data[1]
args.n_output_neurons = dataset.root.stats.train_label[1]
args.n_hidden_neurons = args.n_h


### Learning parameters
if not args.num_samples_train:
    args.num_samples_train = args.dataset.root.stats.train_data[0]

if not args.num_samples_test:
    args.num_samples_test = args.dataset.root.stats.test_data[0]


# Save results and weights
name = r'_' + args.model + r'%d_epochs_nh_%d' % (args.num_samples_train, args.n_hidden_neurons) + args.suffix
args.save_path = os.getcwd() + r'/results/' + args.dataset + name + '.pkl'
args.save_path_weights = os.getcwd() + r'/results/' + args.dataset + name + '_weights.hdf5'

args.ite_test = np.arange(0, args.num_samples_train, args.test_period)

if os.path.exists(args.save_path):
    with open(args.save_path, 'rb') as f:
        args.test_accs = pickle.load(f)
else:
    args.test_accs = {i: [] for i in args.ite_test}


if args.topology_type == 'custom':
    topology = torch.ones([args.n_hidden_neurons + args.n_output_neurons, args.n_input_neurons + args.n_hidden_neurons + args.n_output_neurons], dtype=torch.float)
    # Feel free to fill this with custom topologies

for _ in range(args.num_ite):
    # Create the network
    if args.model == 'wta':
        multivalued_exp.launch_multivalued_exp(args)

    elif args.model == 'binary':
        binary_exp.launch_binary_exp(args)
