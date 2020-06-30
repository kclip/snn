import pickle
from wispike.utils.misc import channel
import torch
import tables
import argparse
from multivalued_snn.utils_multivalued.misc import str2bool
from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary import misc as misc_snn
from utils.filters import get_filter
from binary_snn.utils_binary.misc import refractory_period
import numpy as np

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset', default='mnist_dvs_10_binary')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--model', default='wispike', help='Model type, either "binary" or "wta"')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--labels', nargs='+', default=None, type=int, help='Class labels to be used during training')


    # Arguments common to all models
    parser.add_argument('--n_h', default=256, type=int, help='Number of hidden neurons')
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
    parser.add_argument('--r', default=0.3, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization strength')

    # Arguments for Wispike
    parser.add_argument('--systematic', type=str, default='false', help='Systematic communication')
    parser.add_argument('--snr', type=float, default=None, help='SNR')
    parser.add_argument('--n_output_enc', default=128, type=int, help='')

    args = parser.parse_args()

print(args)

if args.where == 'local':
    home = r'C:/Users/K1804053/PycharmProjects'
elif args.where == 'rosalind':
    home = r'/users/k1804053'
elif args.where == 'jade':
    home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    home = r'/home/k1804053'


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

if args.dataset[:3] == 'shd':
    dataset = home + r'/datasets/shd/' + datasets[args.dataset]
elif args.dataset[:5] == 'mnist':
    dataset = home + r'/datasets/mnist-dvs/' + datasets[args.dataset]
elif args.dataset[:11] == 'dvs_gesture':
    dataset = home + r'/datasets/DvsGesture/' + datasets[args.dataset]
elif args.dataset[:7] == 'swedish':
    dataset = home + r'/datasets/SwedishLeaf_processed/' + datasets[args.dataset]
else:
    print('Error: dataset not found')

args.dataset = tables.open_file(dataset)

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.save_path = None
args.topology = None

### Learning parameters
args.num_samples_test = args.dataset.root.stats.test_data[0]
if args.labels is not None:
    print(args.labels)
    num_samples_test = min(args.num_samples_test, len(misc_snn.find_test_indices_for_labels(args.dataset, args.labels)))
    test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)
else:
    test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

### Network parameters
args.n_input_neurons = args.dataset.root.stats.train_data[1]
args.n_output_neurons = args.dataset.root.stats.train_label[1]
args.n_hidden_neurons = args.n_h


network = SNNetwork(**misc_snn.make_network_parameters(args.n_input_neurons,
                                                       args.n_output_neurons,
                                                       args.n_h,
                                                       args.topology_type,
                                                       args.topology,
                                                       args.density,
                                                       'train',
                                                       args.weights_magnitude,
                                                       args.n_basis_ff,
                                                       get_filter(args.ff_filter),
                                                       args.n_basis_fb,
                                                       get_filter(args.fb_filter),
                                                       args.initialization,
                                                       args.tau_ff,
                                                       args.tau_fb,
                                                       args.mu,
                                                       args.save_path),
                    device=args.device)


weights = r'C:/Users/K1804053/PycharmProjects/results/results_wispike/002__18-06-2020_mnist_dvs_10_binary_binary_20000_epochs_nh_256_nout_128'
network_weights = weights + r'/network_weights.hdf5'

network.import_weights(network_weights)
snr_list = [0, -2, -4, -6, -8, -10]

res = {snr: 0 for snr in snr_list}

for snr in snr_list:
    network.set_mode('test')
    network.reset_internal_state()

    S_prime = args.dataset.root.test.label[:].shape[-1]

    outputs = torch.zeros([len(test_indices), network.n_output_neurons, S_prime])
    loss = 0

    rec = torch.zeros([network.n_learnable_neurons, S_prime])

    for j, sample_idx in enumerate(test_indices):
        refractory_period(network)

        sample = channel(torch.FloatTensor(args.dataset.root.test.data[sample_idx]).to(network.device), network.device, snr)

        for s in range(S_prime):
            log_proba = network(sample[:, s])
            outputs[j, :, s] = network.spiking_history[network.output_neurons, -1]
            rec[:, s] = network.spiking_history[network.learnable_neurons, -1]

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
    print('snr %d, acc %f' % (snr, acc))
    res[snr] = acc

with open(weights + r'/acc_per_snr.npy', 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

