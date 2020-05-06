from __future__ import print_function
import torch
from models.SNN import SNNetwork
# from utils_binary.training_ml_online import train_ml_online, refractory_period
# import utils_binary.training_utils
import data_preprocessing.misc as utils
import numpy as np
import tables
import argparse

''''
Code snippet to train a multivalued SNN.
'''

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--dataset')
    parser.add_argument('--weights')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--kappa', default=0.05, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha softmax coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')

    args = parser.parse_args()


local_data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
# path_to_weights = os.getcwd() + r'/results/'

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

train_shape = tables.open_file(dataset).root.stats.train[:]
test_shape = tables.open_file(dataset).root.stats.test[:]


### Network parameters
if args.dataset[:5] == 'mnist':
    n_input_neurons = 676
elif args.dataset[:11] == 'dvs_gesture':
    n_input_neurons = 1024
if args.dataset[:3] == 'shd':
    n_input_neurons = 700

alphabet_size = train_shape[2]

n_output_neurons = train_shape[1] - n_input_neurons
n_hidden_neurons = 16

mode = 'train_ml_online'

### Learning parameters

learning_rate = args.lr
kappa = args.kappa
alpha = args.alpha
beta = args.beta

### Run training
# Create the network
network = SNNetwork(**utils.make_network_parameters(n_input_neurons, n_output_neurons,
                                                    n_hidden_neurons,
                                                    alphabet_size,
                                                    mode[:8],
                                                    n_basis_ff=8,
                                                    tau_ff=10,
                                                    tau_fb=10,
                                                    mu=1.5,
                                                    dropout_rate=None),
                    temperature=1,
                    device='cpu')


# network.import_weights(path_to_weights + args.weights)
network.import_weights(r'C:\Users\K1804053\PycharmProjects\results\results_multivalued\mnist_dvs_10_train_ml_online9000_epochs_nh_16_no_baseline_no_reg_1_weights.hdf5')

# ### Test accuracy
# test_indices = np.arange(test_shape[0])
test_indices = np.array([0, 200])

network.set_mode('test')
network.reset_internal_state()

S_prime = tables.open_file(dataset).root.stats.test[:][-1]
outputs = torch.zeros([len(test_indices), network.n_output_neurons, network.alphabet_size, S_prime])

hidden_hist = torch.zeros([2, network.n_neurons, network.alphabet_size, S_prime])

spikes = torch.zeros([2, network.n_neurons, network.alphabet_size, S_prime])

for j, sample_idx in enumerate(test_indices):
    utils.refractory_period(network)

    sample = torch.FloatTensor(tables.open_file(dataset).root.test.data[sample_idx])

    for s in range(S_prime):
        log_proba = network(sample[:, :, s])
        outputs[j, :, :, s % S_prime] = network.spiking_history[network.output_neurons, :, -1]
        # hidden_hist[:, :, s] = network.spiking_history[network.hidden_neurons, :, -1]
        spikes[j, :, :, s % S_prime] = network.spiking_history[:, :, -1]

    print(torch.sum(outputs[j], dim=(-1, -2)),
          torch.sum(outputs[j], dim=(-2)),
          torch.max(torch.sum(outputs[j], dim=(-1, -2)), dim=-1).indices + 1,
          torch.max(torch.sum(torch.FloatTensor(tables.open_file(dataset).root.test.label[:][test_indices[j]]), dim=(-1, -2)), dim=-1).indices + 1)

predictions = torch.max(torch.sum(outputs, dim=(-1, -2)), dim=-1).indices
true_classes = torch.max(torch.sum(torch.FloatTensor(tables.open_file(dataset).root.test.label[:][test_indices]), dim=(-1, -2)), dim=-1).indices

acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
np.save('C:/Users/K1804053/PycharmProjects/results/results_multivalued/spikes_no_reg.npy', spikes.numpy())

print('Final test accuracy: %f' % acc)