import numpy as np
import tables
import torch
import argparse
import utils.utils_snn as misc_snn
from utils.utils_wtasnn import str2bool
import wispike.utils.misc as misc_wispike
from wispike.utils import training_utils
from wispike.test import testing_utils
from models.SNN import SNNetwork
import pickle
from misc import mksavedir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--dataset', default='mnist_dvs_10_binary')
    parser.add_argument('--save_path', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--num_ite', default=1, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_train', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--test_period', default=200, type=int, help='')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--start_idx', type=int, default=0, help='When resuming training from existing weights, index to start over from')
    parser.add_argument('--labels', nargs='+', default=None, type=int, help='Class labels to be used during training')
    parser.add_argument('--classifier', type=str, default='snn', choices=['snn', 'mlp'])
    parser.add_argument('--test_type', type=str, default='final', choices=['final', 'per_frame'])
    parser.add_argument('--suffix', type=str, default='', help='Appended to the name of the saved results and weights')

    parser.add_argument('--snr', type=float, default=100, help='SNR')
    parser.add_argument('--n_frames', default=80, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate of classifier')


    # Arguments for VQ-VAE
    parser.add_argument('--embedding_dim', default=32, type=int, help='Size of VQ-VAE latent embeddings')
    parser.add_argument('--num_embeddings', default=10, type=int, help='Number of VQ-VAE latent embeddings')
    parser.add_argument('--lr_vqvae', default=1e-3, type=float, help='Learning rate of VQ-VAE')
    parser.add_argument('--maxiter', default=100, type=int, help='Max number of iteration for BP decoding of LDPC code')


    # Arguments for snn models
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

    args = parser.parse_args()

if args.where == 'local':
    home = r'C:/Users/K1804053/PycharmProjects'
elif args.where == 'rosalind':
    home = r'/users/k1804053'
elif args.where == 'jade':
    home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    home = r'/home/k1804053'

dataset = tables.open_file(home + r'/datasets/mnist-dvs/mnist_dvs_binary_25ms_26pxl_10_digits.hdf5')

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

if not args.num_samples_train:
    args.num_samples_train = dataset.root.stats.train_data[0]
if not args.num_samples_test:
    args.num_samples_test = dataset.root.stats.test_data[0]

args.residual = 80 % args.n_frames
if args.residual:
    args.n_frames += 1

# Make VAE
vqvae, vqvae_optimizer = training_utils.init_vqvae(args, dataset)
args.quantized_dim, args.encodings_dim = misc_wispike.get_intermediate_dims(vqvae, args, dataset)

name = 'vqvae_' + args.classifier + r'_%d_epochs_nh_%d_ny_%d_nframes_%d' % (args.num_samples_train, args.n_h, 2 * np.prod(args.encodings_dim), args.n_frames) + args.suffix
results_path = home + r'/results/'
if args.save_path is None:
    args.save_path = mksavedir(pre=results_path, exp_dir=name)

with open(args.save_path + 'commandline_args.pkl', 'wb') as f:
    pickle.dump(args.__dict__, f, pickle.HIGHEST_PROTOCOL)

args.dataset = dataset

# Make classifier
classifier = training_utils.init_classifier(args)

# LDPC coding
args.H, args.G, args.k = training_utils.init_ldpc(args.encodings_dim)


if args.labels is not None:
    print(args.labels)
    indices = np.random.choice(misc_snn.find_train_indices_for_labels(args.dataset, args.labels), [args.num_samples_train], replace=True)
    num_samples_test = min(args.num_samples_test, len(misc_snn.find_test_indices_for_labels(args.dataset, args.labels)))
    test_indices = np.random.choice(misc_snn.find_test_indices_for_labels(args.dataset, args.labels), [num_samples_test], replace=False)
else:
    indices = np.random.choice(np.arange(args.dataset.root.stats.train_data[0]), [args.num_samples_train], replace=True)
    test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)


args.ite_test = np.arange(0, args.num_samples_train, args.test_period)
args.test_accs = {i: [] for i in args.ite_test}

# Training
train_res_recon_error = []
train_res_perplexity = []

for i, idx in enumerate(indices):
    train_res_recon_error, train_res_perplexity = \
        training_utils.train_vqvae(vqvae, vqvae_optimizer, args, train_res_recon_error, train_res_perplexity, idx)
    training_utils.train_classifier(classifier, args, idx)

    if (i + 1) % args.test_period == 0:
        print('Testing at step %d...' % (i + 1))
        acc, _ = testing_utils.get_acc_classifier(classifier, vqvae, args, test_indices)
        print('test accuracy: %f' % acc)
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-args.test_period:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-args.test_period:]))

        args.test_accs[int(i + 1)].append(acc)
        with open(args.save_path + r'/test_accs.pkl', 'wb') as f:
            pickle.dump(args.test_accs, f, pickle.HIGHEST_PROTOCOL)
        if isinstance(classifier, SNNetwork):
            classifier.save(args.save_path + r'/snn_weights.hdf5')
        else:
            torch.save(classifier.state_dict(), args.save_path + r'mlp_weights.pt')
        torch.save(vqvae.state_dict(), args.save_path + r'vqvae_weights.pt')

