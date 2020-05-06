from __future__ import print_function
import torch
from SNN_old import SNNetwork
from utils.training_utils import train_ml_online, train_policy_based_rl_online, get_acc_and_loss, train_ml_batch, train_rl_batch
import utils.training_utils
import time
import numpy as np
import tables
import math
import argparse
import os

''''
Code snippet to train a multivalued SNN.
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--n_hidden', default=1, type=int, help='')
    parser.add_argument('--kappa', default=0.05, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--density', default=1., type=float, help='Topology density')
    parser.add_argument('--gain_fb', default=0., type=float, help='')
    parser.add_argument('--gain_ff', default=0., type=float, help='')
    parser.add_argument('--gain_bias', default=0., type=float, help='')

    args = parser.parse_args()


distant_data_path = r'/users/k1804053/FL-SNN-multivalued/'
local_data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
save_path = os.getcwd() + r'/results'

dataset = local_data_path + r'sc/sc_sod_test.hdf5'


input_train = torch.FloatTensor(tables.open_file(dataset).root.train.data[:])
output_train = torch.FloatTensor(tables.open_file(dataset).root.train.label[:])

input_test = torch.FloatTensor(tables.open_file(dataset).root.test.data[:])
output_test = torch.FloatTensor(tables.open_file(dataset).root.test.label[:])


### Network parameters
n_input_neurons = input_train.shape[1]
n_output_neurons = output_train.shape[1]
n_hidden_neurons = args.n_hidden_neurons
alphabet_size = input_train.shape[-2]
mode = args.mode

### Learning parameters
epochs = input_train.shape[0]
epochs_test = input_test.shape[0]

learning_rate = args.lr
kappa = args.kappa
alpha = 3
beta = args.beta
num_ite = args.num_ite

### Randomly select training samples
indices = np.random.choice(np.arange(input_train.shape[0]), [epochs], replace=False)

S_prime = input_train.shape[-1]
S = epochs * S_prime

### Run training
# Create the network
network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode[:8], topology_type='sparse',
                                                                   density=args.density, gain_fb=args.gain_fb, gain_ff=args.gain_ff, gain_bias=args.gain_bias,
                                                                   initialization='glorot'))
# Train it
train_policy_based_rl_online(network, input_train[indices], output_train[indices], learning_rate, kappa, alpha, beta)

# ### Test accuracy
test_indices = np.random.choice(np.arange(input_test.shape[0]), [epochs_test], replace=False)
acc, loss = get_acc_and_loss(network, input_test[test_indices], output_test[test_indices])

print(acc)
