from __future__ import print_function
import torch
from SNN import SNNetwork
from utils.training_utils import train_ml_online
from utils.training_utils import get_acc_and_loss
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
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--kappa', default=0.2, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha softmax coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--beta_2', default=0.999, type=float)
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization factor')
    parser.add_argument('--r', default=0.8, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--suffix', type=str, default='', help='')


    args = parser.parse_args()


distant_data_path = r'/users/k1804053/FL-SNN-multivalued/'
local_data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
save_path = os.getcwd() + r'/results'

dataset_0_4 = r'mnist_dvs_25ms_26pxl_0_4.hdf5'
dataset_5_9 = r'mnist_dvs_25ms_26pxl_5_9.hdf5'


if args.where == 'local':
    dataset_0_4 = local_data_path + r'/mnist-dvs/' + dataset_0_4
    dataset_5_9 = local_data_path + r'/mnist-dvs/' + dataset_5_9

elif args.where == 'distant':
    dataset_0_4 = distant_data_path + dataset_0_4
    dataset_5_9 = distant_data_path + dataset_5_9

elif args.where == 'gcloud':
    dataset_0_4 = r'/home/k1804053/' + dataset_0_4
    dataset_5_9 = r'/home/k1804053/' + dataset_5_9

train_shape_0_4 = tables.open_file(dataset_0_4).root.stats.train[:]
test_shape_0_4 = tables.open_file(dataset_0_4).root.stats.test[:]

train_shape_5_9 = tables.open_file(dataset_5_9).root.stats.train[:]
test_shape_5_9 = tables.open_file(dataset_5_9).root.stats.test[:]

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print(args.disable_cuda)

### Network parameters
n_input_neurons = 676
alphabet_size = 2
mode = 'train_ml_online'

n_output_neurons = 5
n_hidden_neurons = 128

### Learning parameters
learning_rate = args.lr / n_hidden_neurons
kappa = args.kappa
alpha = args.alpha
beta = args.beta
beta_2 = args.beta_2
gamma = args.gamma
r = args.r


# Test parameters
save_path = os.getcwd() + r'/results/forgetting_experience_nh_128.pkl'
save_path_weights = os.getcwd() + r'/results/forgetting_experience_weights.hdf5'

indices_0_4 = np.random.choice(np.arange(train_shape_0_4[0]), [5000], replace=True)
indices_5_9 = [np.random.choice(np.arange(train_shape_5_9[0]), [10], replace=True),
               np.random.choice(np.arange(train_shape_5_9[0]), [90], replace=True),
               np.random.choice(np.arange(train_shape_5_9[0]), [400], replace=True),
               np.random.choice(np.arange(train_shape_5_9[0]), [500], replace=True),
               np.random.choice(np.arange(train_shape_5_9[0]), [1000], replace=True),
               np.random.choice(np.arange(train_shape_5_9[0]), [1000], replace=True),
               np.random.choice(np.arange(train_shape_5_9[0]), [1000], replace=True),
               np.random.choice(np.arange(train_shape_5_9[0]), [1000], replace=True)]

network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode[:8]), device=args.device)

# Train it
t0 = time.time()

test_indices_0_4 = np.random.choice(np.arange(test_shape_0_4[0]), [test_shape_0_4[0]], replace=False)
test_indices_5_9 = np.random.choice(np.arange(test_shape_5_9[0]), [test_shape_5_9[0]], replace=False)

res = {'acc_0_4': [], 'acc_5_9': []}

print(args)

acc_0_4, _ = get_acc_and_loss(network, tables.open_file(dataset_0_4), test_indices_0_4)
acc_5_9, _ = get_acc_and_loss(network, tables.open_file(dataset_5_9), test_indices_5_9)
print('Before training: Acc on 0-4: %f, Acc on 5-9: %f' % (acc_0_4, acc_5_9))

res['acc_0_4'].append(acc_0_4)
res['acc_0_4'].append(acc_5_9)
train_ml_online(network, dataset_0_4, indices_0_4, None, None, learning_rate, kappa, beta, gamma, r, save_path, save_path_weights)

acc_0_4, _ = get_acc_and_loss(network, tables.open_file(dataset_0_4), test_indices_0_4)
acc_5_9, _ = get_acc_and_loss(network, tables.open_file(dataset_5_9), test_indices_5_9)
print('After 5000 training iterations on 0-4: Acc on 0-4: %f, Acc on 5-9: %f' % (acc_0_4, acc_5_9))

res['acc_0_4'].append(acc_0_4)
res['acc_0_4'].append(acc_5_9)

num_training_ite_5_9 = 0

for indices in indices_5_9:
    train_ml_online(network, dataset_5_9, indices, None, None, learning_rate, kappa, beta, gamma, r, save_path, save_path_weights)

    num_training_ite_5_9 += len(indices)

    acc_0_4, _ = get_acc_and_loss(network, tables.open_file(dataset_0_4), test_indices_0_4)
    acc_5_9, _ = get_acc_and_loss(network, tables.open_file(dataset_5_9), test_indices_5_9)
    print('After %d training iterations on 0-4: Acc on 0-4: %f, Acc on 5-9: %f' % (num_training_ite_5_9, acc_0_4, acc_5_9))

    res['acc_0_4'].append(acc_0_4)
    res['acc_0_4'].append(acc_5_9)

with open(save_path, 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
