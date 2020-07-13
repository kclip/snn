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
from wispike.utils import training_utils, testing_utils
from wispike.utils.misc import channel_coding_decoding, channel, framed_to_example, example_to_framed, binarize, get_intermediate_dims


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

    parser.add_argument('--embedding_dim', default=32, type=int, help='Size of VQ-VAE latent embeddings')
    parser.add_argument('--num_embeddings', default=10, type=int, help='Number of VQ-VAE latent embeddings')
    parser.add_argument('--lr_vqvae', default=1e-3, type=float, help='Learning rate of VQ-VAE')
    parser.add_argument('--maxiter', default=100, type=int, help='Max number of iteration for BP decoding of LDPC code')


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

args.dataset = tables.open_file(home + r'/datasets/mnist-dvs/mnist_dvs_binary_25ms_26pxl_10_digits.hdf5')

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
args.n_frames = 80

args.residual = 80 % args.n_frames
if args.residual:
    args.n_frames += 1

### Encoder & classifier
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

vqvae, _ = training_utils.init_vqvae(args)

weights = r'C:\Users\K1804053\PycharmProjects\results\results_wispike\017__06-07-2020_vqvae_snn_100000_epochs_nh_512_nout_360_snr_m_4'
network_weights = weights + r'/snn_weights.hdf5'
vqvae_weights = weights + r'/vqvae_weights.pt'

network.import_weights(network_weights)
vqvae.load_state_dict(torch.load(vqvae_weights))
network.set_mode('test')
vqvae.eval()


### Channel & coding
args.quantized_dim, args.encodings_dim = get_intermediate_dims(vqvae, args)
args.H, args.G, args.k = training_utils.init_ldpc(args.encodings_dim)


snr_list = [0, -2, -4, -6, -8, -10]
# snr_list = [0]

res_final = {snr: 0 for snr in snr_list}
res_pf = {snr: 0 for snr in snr_list}


for snr in snr_list:
    args.snr = snr
    network.reset_internal_state()

    predictions_final = torch.zeros([len(test_indices)], dtype=torch.long)
    predictions_pf = torch.zeros([len(test_indices), args.n_frames], dtype=torch.long)

    T = args.dataset.root.test.label[:].shape[-1]

    outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])

    for i, idx in enumerate(test_indices):
        data = example_to_framed(args.dataset.root.test.data[idx, :, :], args)
        data_reconstructed = torch.zeros(data.shape)

        for t in range(T):
            frame = data[t].unsqueeze(0)
            with torch.autograd.no_grad():
                _, encodings = vqvae.encode(frame)
                encodings_decoded = channel_coding_decoding(args, encodings)
                data_reconstructed[t] = vqvae.decode(encodings_decoded, args.quantized_dim)

        predictions_final[i], predictions_pf[i] = testing_utils.classify(network, data_reconstructed, args, 'both')

    true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices

    accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
    accs_per_frame = torch.zeros([T], dtype=torch.float)
    for t in range(1, T):
        acc = float(torch.sum(predictions_pf[:, t] == true_classes, dtype=torch.float) / len(predictions_pf))
        accs_per_frame[t] = acc

    print('snr %d, acc %f' % (snr, acc))
    res_final[snr] = accs_final
    res_pf[snr] = accs_per_frame

with open(weights + r'/acc_per_snr_final.npy', 'wb') as f:
    pickle.dump(res_final, f, pickle.HIGHEST_PROTOCOL)

with open(weights + r'/acc_per_snr_per_frame.npy', 'wb') as f:
    pickle.dump(res_pf, f, pickle.HIGHEST_PROTOCOL)
