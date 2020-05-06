from __future__ import print_function
import torch
from SNN import SNNetwork
from utils.training_utils import train_ml_online, get_acc_and_loss
import utils.filters as filters
import utils.training_utils
import time
import numpy as np
import tables
import pickle
import argparse
import os

''''
Code snippet to train a multivalued SNN.
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset')
    parser.add_argument('--mode', default='train_ml_online', help='Feedforward or interactive readout')
    parser.add_argument('--num_ite', default=1, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--kappa', default=0.2, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha softmax coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--beta_2', default=0.999, type=float)
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization factor')
    parser.add_argument('--r', default=0.6, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--disable-cuda', default=True, help='Disable CUDA')


    args = parser.parse_args()


distant_data_path = r'/users/k1804053/FL-SNN-multivalued/'
local_data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
save_path = os.getcwd() + r'/results'

datasets = {
            'swedish_3_009': r'swedish_sdata_send-on-delta_G0.090_C3_class3_memorization.hdf5',
            'swedish_3_007': r'swedish_sdata_send-on-delta_G0.070_C3_class3_memorization.hdf5',
            'swedish_2_009': r'swedish_sdata_send-on-delta_G0.090_C2_class3_memorization.hdf5',
            'swedish_2_007': r'swedish_sdata_send-on-delta_G0.070_C2_class3_memorization.hdf5'
            }


if args.where == 'local':
    dataset = local_data_path + r'/SwedishLeaf_processed/' + datasets[args.dataset]

elif args.where == 'distant':
    dataset = distant_data_path + datasets[args.dataset]

elif args.where == 'gcloud':
    dataset = r'/home/k1804053/' + datasets[args.dataset]


train_shape = tables.open_file(dataset).root.train.data[:].shape

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


n_input_neurons = 20
n_output_neurons = 1
n_hidden_neurons = 16

alphabet_size = train_shape[2]
mode = args.mode

### Learning parameters
epochs = 50
learning_rate = args.lr / n_hidden_neurons
kappa = args.kappa
alpha = args.alpha
beta = args.beta
beta_2 = args.beta_2
gamma = args.gamma
r = args.r
num_ite = args.num_ite

# Test parameters
ite_test = []

name = r'memorization_ml_online_%d_epochs_nh_%d' % (epochs, n_hidden_neurons)
save_path = os.getcwd() + r'/results/' + args.dataset + name + '.pkl'

indices = [0] * epochs

# for _ in range(num_ite):
# Create the network

t0 = time.time()

network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons,
                                                                   n_hidden_neurons, alphabet_size, mode[:8], n_basis_ff=8, tau_ff=10, tau_fb=10,
                                                                   ff_filter=filters.alpha_function, fb_filter=filters.alpha_function, mu=0.75)).to(args.device)

# Train it
train_ml_online(network, dataset, indices, None, ite_test, learning_rate, kappa, beta, gamma, r, args.device, None)

print('time: %f' % (time.time() - t0))

### Test accuracy

network.set_mode('test')
network.reset_internal_state()

S_prime = tables.open_file(dataset).root.stats.train[:][-1]
outputs = torch.zeros([1, network.n_output_neurons, network.alphabet_size, S_prime])

loss = 0

sample = torch.FloatTensor(tables.open_file(dataset).root.train.data[0]).to(args.device)

for s in range(S_prime):
    log_proba = network(sample[:, :, s])
    loss += torch.sum(log_proba).numpy()
    outputs[0, :, :, s % S_prime] = network.spiking_history[network.output_neurons, :, -1]

true_output = torch.FloatTensor(tables.open_file(dataset).root.train.label[:]).to(args.device)

print(outputs[0, 0, :, :10])
print(true_output[0, 0, :, :10])
print((outputs == true_output)[0, 0, :, :10])
acc = torch.sum(outputs == true_output, dtype=torch.float) / (true_output.shape[-2] * true_output.shape[-1])

print('Final test accuracy: %f' % acc)

np.save(r'C:\Users\K1804053\Desktop\PhD\Notebooks\test_memorization.npy', outputs.numpy())
# with open(save_path, 'wb') as f:

#     pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)

# np.save(save_path + '/acc_' + args.dataset + args.mode + '_%d_epochs' + '_nh_%d' + '.npy' % (args.epochs, n_hidden_neurons), test_accs)
