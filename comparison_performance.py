import torch
import tables
import argparse
from multivalued_snn.utils_multivalued.misc import str2bool
from binary_snn.utils_binary import misc as misc_snn
from wispike.test.lzma_ldpc import lzma_test
from wispike.test.ook import ook_test
from wispike.test.vqvae_ldpc import vqvae_test
from wispike.test.wispike_test import wispike_test

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--model', choices=['wispike', 'ook', 'vqvae', 'lzma'])
    parser.add_argument('--dataset', default='mnist_dvs_10_binary')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--num_ite', default=1, type=int)

    parser.add_argument('--classifier', type=str, default='snn', choices=['snn', 'mlp'])
    parser.add_argument('--maxiter', default=100, type=int, help='Max number of iteration for BP decoding of LDPC code')
    parser.add_argument('--n_frames', default=80, type=int)

    # Arguments common to all models
    parser.add_argument('--n_h', default=256, type=int, help='Number of hidden neurons')

    # Arguments for Wispike
    parser.add_argument('--systematic', type=str, default='false', help='Systematic communication')
    parser.add_argument('--snr', type=float, default=None, help='SNR')
    parser.add_argument('--n_output_enc', default=128, type=int, help='')

    args = parser.parse_args()

print(args)

if args.where == 'local':
    args.home = r'C:/Users/K1804053/PycharmProjects'
elif args.where == 'rosalind':
    args.home = r'/users/k1804053'
elif args.where == 'jade':
    args.home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    args.home = r'/home/k1804053'


dataset = args.home + r'/datasets/mnist-dvs/mnist_dvs_binary_25ms_26pxl_10_digits.hdf5'

args.dataset = tables.open_file(dataset)

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.save_path = None

### Learning parameters
args.num_samples_test = args.dataset.root.stats.test_data[0]
args.labels = [1, 7]
args.num_samples_test = min(args.num_samples_test, len(misc_snn.find_test_indices_for_labels(args.dataset, args.labels)))

### Network parameters
args.n_input_neurons = args.dataset.root.stats.train_data[1]
args.n_output_neurons = args.dataset.root.stats.train_label[1]
args.n_hidden_neurons = args.n_h


if args.model == 'wispike':
    wispike_test(args)

elif args.model == 'ook':
    ook_test(args)

elif args.model == 'vqvae':
    vqvae_test(args)

elif args.model == 'lzma':
    lzma_test(args)
