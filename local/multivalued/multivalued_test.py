from __future__ import print_function
import torch
from models.SNN import SNNetwork
from utils import training_ml_online, training_ml_batch, training_rl_online, training_rl_batch
import data_preprocessing.misc as utils
import time
import numpy as np
import tables
import pickle
import argparse
import os

''''
Code snippet to train a multivalued SNN.
'''

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--mode', default='train_ml_online', help='Feedforward or interactive readout')
    parser.add_argument('--num_ite', default=5, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--n_h', default=128, type=int, help='Number of hidden neurons')
    parser.add_argument('--mu', default=1.5, type=float, help='')
    parser.add_argument('--tau_ff', default=10, type=int, help='')
    parser.add_argument('--n_basis_ff', default=8, type=int, help='')
    parser.add_argument('--tau_fb', default=10, type=int, help='')
    parser.add_argument('--dropout_rate', default=None, type=float, help='')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--kappa', default=0.2, type=float, help='eligibility trace decay coefficient')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha softmax coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--beta_2', default=0.999, type=float)
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization factor')
    parser.add_argument('--r', default=0.8, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--suffix', type=str, default='', help='')
    parser.add_argument('--labels', nargs='+', default=None, type=int)
    parser.add_argument('--T', type=float, default=1., help='temperature')

    args = parser.parse_args()


distant_data_path = r'/users/k1804053/snn/multivalued_snn/'
local_data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
save_path = os.getcwd() + r'/results'

datasets = {'mnist_dvs_2': r'mnist_dvs_25ms_26pxl_2_digits_polarity.hdf5',
            'mnist_dvs_10_binary': r'mnist_dvs_binary_25ms_26pxl_10_digits.hdf5',
            'mnist_dvs_10': r'mnist_dvs_25ms_26pxl_10_digits_polarity.hdf5',
            'mnist_dvs_10_c_3': r'mnist_dvs_25ms_26pxl_10_digits_C_3.hdf5',
            'mnist_dvs_10_c_5': r'mnist_dvs_25ms_26pxl_10_digits_C_5.hdf5',
            'mnist_dvs_10_c_7': r'mnist_dvs_25ms_26pxl_10_digits_C_7.hdf5',
            'mnist_dvs_10ms_polarity': r'mnist_dvs_10ms_26pxl_10_digits_polarity.hdf5',
            'dvs_gesture_5ms': r'dvs_gesture_5ms_11_classes.hdf5',
            'dvs_gesture_5ms_5_classes': r'dvs_gesture_5ms_5_classes.hdf5',
            'dvs_gesture_20ms_2_classes': r'dvs_gesture_20ms_2_classes.hdf5',
            'dvs_gesture_5ms_2_classes': r'dvs_gesture_5ms_2_classes.hdf5',
            'dvs_gesture_5ms_3_classes': r'dvs_gesture_5ms_3_classes.hdf5',
            'dvs_gesture_15ms': r'dvs_gesture_15ms_11_classes.hdf5',
            'dvs_gesture_20ms': r'dvs_gesture_20ms_11_classes.hdf5',
            'dvs_gesture_30ms': r'dvs_gesture_30ms_11_classes.hdf5',
            'dvs_gesture_20ms_5_classes': r'dvs_gesture_20ms_5_classes.hdf5',
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

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print(args.disable_cuda)

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
n_hidden_neurons = args.n_h

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

learning_rate = args.lr / max(1, min(n_hidden_neurons, 1024))
kappa = args.kappa
alpha = args.alpha
beta = args.beta
beta_2 = args.beta_2
gamma = args.gamma
r = args.r
num_ite = args.num_ite
test_period = args.test_period

# Test parameters
ite_test = np.arange(0, num_samples_train, test_period)

name = r'_' + mode + r'%d_epochs_nh_%d' % (num_samples_train, n_hidden_neurons) + args.suffix
save_path = os.getcwd() + r'/results/' + args.dataset + name + '.pkl'
save_path_weights = os.getcwd() + r'/results/' + args.dataset + name + '_weights.hdf5'

if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
        test_accs = pickle.load(f)
else:
    test_accs = {i: [] for i in ite_test}

### Find indices
if args.labels is not None:
    print(args.labels)
    indices = np.random.choice(utils.find_train_indices_for_labels(dataset, args.labels), [num_samples_train], replace=True)
    num_samples_test = min(num_samples_test, len(utils.find_test_indices_for_labels(dataset, args.labels)))
    test_indices = np.random.choice(utils.find_test_indices_for_labels(dataset, args.labels), [num_samples_test], replace=False)

else:
    indices = np.random.choice(np.arange(train_shape[0]), [num_samples_train], replace=True)
    test_indices = np.random.choice(np.arange(test_shape[0]), [num_samples_test], replace=False)

# Create the network
network = SNNetwork(**utils.make_network_parameters(n_input_neurons, n_output_neurons,
                                                    n_hidden_neurons,
                                                    alphabet_size,
                                                    mode[:8],
                                                    n_basis_ff=args.n_basis_ff,
                                                    tau_ff=args.tau_ff,
                                                    tau_fb=args.tau_fb,
                                                    mu=args.mu,
                                                    dropout_rate=args.dropout_rate),
                    temperature=args.T,
                    device=args.device)

print(network.get_parameters()['ff_weights'].shape)

if args.weights is not None:
    network.import_weights(args.weights)
# Train it
t0 = time.time()

print(args)

if mode == 'train_ml_online':
    test_accs = training_ml_online.train(network, dataset, indices, test_indices, test_accs, learning_rate, kappa, beta, gamma, r, args.start_idx, save_path, save_path_weights)
elif mode == 'train_rl_online':
    training_rl_online.train(network, input_train, output_train, indices, learning_rate, kappa, alpha, beta, gamma, r)
if mode == 'train_ml_batch':
    training_ml_batch.train(network, input_train, output_train, indices, learning_rate, beta, gamma, r)
elif mode == 'train_rl_batch':
    training_rl_batch.train(network, input_train[indices], output_train[indices], learning_rate, alpha, beta, gamma, r)


print('Number of samples trained on: %d, time: %f' % (num_samples_train, time.time() - t0))
