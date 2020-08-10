import torch
import tables
import argparse
from multivalued_snn.utils_multivalued.misc import str2bool
from binary_snn.utils_binary import misc as misc_snn
from wispike.test.lzma_ldpc import lzma_test
from wispike.test.ook import ook_test, ook_ldpc_test
from wispike.test.vqvae_ldpc import vqvae_test
from wispike.test.wispike_test import wispike_test
import pickle

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--model', choices=['wispike', 'ook', 'ook_ldpc', 'vqvae', 'lzma'])
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--num_ite', default=1, type=int)
    parser.add_argument('--snr_list', nargs='+', default=None, type=int, help='')

    parser.add_argument('--classifier', type=str, default='snn', choices=['snn', 'mlp'])
    parser.add_argument('--classifier_weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--maxiter', default=100, type=int, help='Max number of iteration for BP decoding of LDPC code')
    parser.add_argument('--ldpc_rate', default=2, type=float)

    args = parser.parse_args()

print(args)

if args.where == 'local':
    args.home = r'C:/Users/K1804053/PycharmProjects'
    args.results = args.home + '/results/results_wispike/'
elif args.where == 'rosalind':
    args.home = r'/users/k1804053'
    args.results = args.home + '/results/'
elif args.where == 'jade':
    args.home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    args.home = r'/home/k1804053'


try:
    exp_args_path = args.results + args.weights + '/commandline_args.pkl'
    args_dict = vars(args)

    with open(exp_args_path, 'rb') as f:
        exp_args = pickle.load(f)

    new_keys = [k for k in list(exp_args.keys()) if k not in list(args_dict.keys())]
    for key in new_keys:
        args_dict[key] = exp_args[key]
except TypeError:
    pass


dataset = args.home + r'/datasets/mnist-dvs/mnist_dvs_binary_25ms_26pxl_10_digits.hdf5'
args.dataset = tables.open_file(dataset)

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.save_path = None

args.labels = [1, 7]

### Learning parameters
args.num_samples_test = args.dataset.root.stats.test_data[0]
args.num_samples_test = min(args.num_samples_test, len(misc_snn.find_test_indices_for_labels(args.dataset, args.labels)))

### Network parameters
args.n_input_neurons = args.dataset.root.stats.train_data[1]
args.n_output_neurons = args.dataset.root.stats.train_label[1]

if 'n_h' not in vars(args):
    args.n_h = 256
args.n_hidden_neurons = args.n_h

if args.model == 'wispike':
    wispike_test(args)

elif args.model == 'ook':
    ook_test(args)

elif args.model == 'ook_ldpc':
    ook_ldpc_test(args)

elif args.model == 'vqvae':
    vqvae_test(args)

elif args.model == 'lzma':
    lzma_test(args)
