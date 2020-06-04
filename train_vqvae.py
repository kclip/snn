import numpy as np
import tables
from wispike.models.vqvae import Model
import torch
import argparse
import os
import binary_snn.utils_binary.misc as misc
from multivalued_snn.utils_multivalued.misc import str2bool
import torch.optim as optim
import pyldpc
from wispike.utils.misc import test
from wispike.utils.training_utils import train_classifier, train_vqvae
from binary_snn.models.SNN import SNNetwork
from utils.filters import get_filter
from wispike.models.mlp import MLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset', default='mnist_dvs_10_binary')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--model', default='binary', choices=['binary', 'wta', 'wispike'], help='Model type, either "binary" or "wta"')
    parser.add_argument('--num_ite', default=5, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_train', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--test_period', default=1000, type=int, help='')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--start_idx', type=int, default=0, help='When resuming training from existing weights, index to start over from')
    parser.add_argument('--suffix', type=str, default='', help='Appended to the name of the saved results and weights')
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
    parser.add_argument('--r', default=0.8, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization strength')

    # Arguments for Wispike
    parser.add_argument('--systematic', type=str, default='true', help='Systematic communication')
    parser.add_argument('--snr', type=float, default=None, help='SNR')
    parser.add_argument('--n_output_enc', default=128, type=int, help='')
    # parser.add_argument('--beta', type=float, default=1.0,
    #     help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    args = parser.parse_args()


if args.where == 'local':
    data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
elif args.where == 'distant':
    data_path = r'/users/k1804053/datasets/'
elif args.where == 'gcloud':
    data_path = r'/home/k1804053/datasets/'

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

if args.dataset[:3] == 'shd':
    dataset = data_path + r'/shd/' + datasets[args.dataset]
elif args.dataset[:5] == 'mnist':
    dataset = data_path + r'/mnist-dvs/' + datasets[args.dataset]
elif args.dataset[:11] == 'dvs_gesture':
    dataset = data_path + r'/DvsGesture/' + datasets[args.dataset]
elif args.dataset[:7] == 'swedish':
    dataset = data_path + r'/SwedishLeaf_processed/' + datasets[args.dataset]
else:
    print('Error: dataset not found')


args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.dataset = tables.open_file(dataset)

# Make VAE
args.n_frames = 1
residual = 80 % args.n_frames
if residual:
    args.n_frames += 1

num_input_channels = 80 // args.n_frames

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 32
num_embeddings = 12

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

vqvae = Model(num_input_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(args.device)
optimizer = optim.Adam(vqvae.parameters(), lr=learning_rate, amsgrad=False)


# Make classifier
if args.classifier == 'snn':
    n_input_neurons = args.dataset.root.stats.train_data[1]
    n_output_neurons = args.dataset.root.stats.train_label[1]

    classifier = SNNetwork(**misc.make_network_parameters(n_input_neurons,
                                                          n_output_neurons,
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

if args.classifier == 'mlp':
    n_input_neurons = np.prod(args.dataset.root.stats.train_data[1:])
    n_output_neurons = args.dataset.root.stats.train_label[1]

    classifier = MLP(args.n_input_neurons, args.n_h, n_output_neurons)


# LDPC coding
ldpc_codewords_length = 676
d_v = 3
d_c = 4
snr = 1000000

# Make LDPC
H, G = pyldpc.make_ldpc(ldpc_codewords_length, d_v, d_c, systematic=True, sparse=True)
_, k = G.shape

if not args.num_samples_train:
    args.num_samples_train = args.dataset.root.stats.train_data[0]

if not args.num_samples_test:
    args.num_samples_test = args.dataset.root.stats.test_data[0]

if args.labels is not None:
    print(args.labels)
    indices = np.random.choice(misc.find_train_indices_for_labels(args.dataset, args.labels), [args.num_samples_train], replace=True)
    num_samples_test = min(args.num_samples_test, len(misc.find_test_indices_for_labels(args.dataset, args.labels)))
    test_indices = np.random.choice(misc.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)
else:
    indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True)
    test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)

best_loss = -1.

# Training
vqvae.train()
train_res_recon_error = []
train_res_perplexity = []
for i, sample_idx in enumerate(indices):
    train_vqvae(vqvae, optimizer, args, train_res_recon_error, train_res_perplexity, sample_idx)
    train_classifier(classifier, optimizer, args, sample_idx, weights=None)

    if (i + 1) % args.test_period == 0:
        acc = test(classifier, vqvae, args, test_indices)
        print('test accuracy at ite %d: %f' % (int(i + 1), acc))

    if (i + 1) % 100 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()
