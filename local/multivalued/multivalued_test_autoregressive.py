from __future__ import print_function
import torch
from SNN_autoregressive import SNNetwork
from utils.training_utils import train_ml_online, train_policy_based_rl_online, get_acc_and_loss, train_ml_batch, train_rl_batch
import utils.training_utils
import time
import numpy as np
import tables
import pickle
import argparse
import os
from utils import filters

''''
Code snippet to train a multivalued SNN.
'''


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where')
    parser.add_argument('--dataset')
    parser.add_argument('--mode', help='Feedforward or interactive readout')
    parser.add_argument('--num_ite', default=5, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--dt', type=int, help='Sampling rate in ms')
    parser.add_argument('--tau_syn', type=int, help='Synaptic exponential constant')
    parser.add_argument('--tau_ff', type=int, help='Feedforward exponential constant')
    parser.add_argument('--tau_fb', type=int, help='Feedback exponential constant')

    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--kappa', default=0.2, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha softmax coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--beta_2', default=0.999, type=float)
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization factor')
    parser.add_argument('--r', default=0.8, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--disable-cuda', type=bool, default=True, help='Disable CUDA')


    args = parser.parse_args()


distant_data_path = r'/users/k1804053/FL-SNN-multivalued/'
local_data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
save_path = os.getcwd() + r'/results'

datasets = {'mnist_dvs_2': r'mnist_dvs_25ms_26pxl_2_digits_polarity.hdf5',
            'mnist_dvs_2_binary': r'mnist_dvs_25ms_26pxl_2_digits_binary.hdf5',
            'mnist_dvs_10': r'mnist_dvs_25ms_26pxl_10_digits_polarity.hdf5',
            'mnist_dvs_10_c_3': r'mnist_dvs_25ms_26pxl_10_digits_C_3.hdf5',
            'mnist_dvs_10_c_5': r'mnist_dvs_25ms_26pxl_10_digits_C_5.hdf5',
            'mnist_dvs_10_c_7': r'mnist_dvs_25ms_26pxl_10_digits_C_7.hdf5',
            'mnist_dvs_10ms_polarity': r'mnist_dvs_10ms_26pxl_10_digits_polarity.hdf5',
            'dvs_gesture_5ms': r'dvs_gesture_5ms_11_classes.hdf5',
            'dvs_gesture_20ms': r'dvs_gesture_20ms_11_classes.hdf5',
            'dvs_gesture_1ms': r'dvs_gesture_1ms_11_classes.hdf5',
            'shd_eng_c_2': r'shd_10ms_10_classes_eng_C_2.hdf5',
            'shd_all_c_2': r'shd_10ms_10_classes_all_C_2.hdf5'
            }


if args.where == 'local':
    if args.dataset[:3] == 'shd':
        dataset = local_data_path + r'/shd/' + datasets[args.dataset]
    elif args.dataset[:5] == 'mnist':
        dataset = local_data_path + r'/mnist-dvs/' + datasets[args.dataset]
    elif args.dataset[:11] == 'dvs_gesture':
        dataset = local_data_path + r'/DvsGesture/' + datasets[args.dataset]
    elif args.dataset[:7] == 'swedish':
        dataset = local_data_path + r'/SwedishLeaf_processed/' + datasets[args.dataset]
    else:
        print('Error: dataset not found')

elif args.where == 'distant':
    dataset = distant_data_path + datasets[args.dataset]
elif args.where == 'gcloud':
    dataset = r'/home/k1804053/' + datasets[args.dataset]

train_shape = tables.open_file(dataset).root.stats.train[:]
test_shape = tables.open_file(dataset).root.stats.test[:]

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


### Network parameters
if args.dataset[:5] == 'mnist':
    n_input_neurons = 676
elif args.dataset[:11] == 'dvs_gesture':
    n_input_neurons = 1024
if args.dataset[:3] == 'shd':
    n_input_neurons = 700

alphabet_size = train_shape[2]
mode = args.mode

n_output_neurons = train_shape[1] - n_input_neurons
n_hidden_neurons = 256

### Learning parameters
if args.epochs:
    num_samples_train = int(args.epochs * train_shape[0])
elif args.num_samples:
    num_samples_train = args.num_samples
else:
    num_samples_train = train_shape[0]

if args.num_samples_test:
    num_samples_test = args.num_samples_test
else:
    num_samples_test = test_shape[0]

learning_rate = args.lr / n_hidden_neurons
kappa = args.kappa
alpha = args.alpha
beta = args.beta
beta_2 = args.beta_2
gamma = args.gamma
r = args.r
num_ite = args.num_ite


# Test parameters
ite_test = [500, 1000, 5000, 9000, 18000, 27000, 36000, 45000, 54000, 60000, 75000, 90000]

name = r'_ml_online_%d_epochs_nh_%d_autoregressive' % (num_samples_train, n_hidden_neurons)
save_path = os.getcwd() + r'/results/' + args.dataset + name + '.pkl'

indices = np.random.choice(np.arange(train_shape[0]), [num_samples_train], replace=True)

# for _ in range(num_ite):
# Create the network
network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode[:8],
                                                                   tau_ff=args.tau_ff, tau_fb=args.tau_fb), dt=args.dt, tau_syn=args.tau_syn)

# Train it
t0 = time.time()

test_indices = np.random.choice(np.arange(test_shape[0]), [num_samples_test], replace=False)

if mode == 'train_ml_online':
    test_accs = train_ml_online(network, dataset, indices, test_indices, ite_test, learning_rate, kappa, beta, gamma, r, save_path)


print('Number of samples trained on: %d, time: %f' % (num_samples_train, time.time() - t0))

# ### Test accuracy

# acc, loss = get_acc_and_loss(network, input_test[test_indices], output_test[test_indices])

# test_accs[epochs].append(acc)
# test_accs.append(acc)

# print('Final test accuracy: %f' % acc)

# with open(save_path, 'wb') as f:
#     pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)

# np.save(save_path + '/acc_' + args.dataset + args.mode + '_%d_epochs' + '_nh_%d' + '.npy' % (args.epochs, n_hidden_neurons), test_accs)
