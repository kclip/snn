from models.SNN import SNNetwork
from utils import utils_wispike
from data_preprocessing import misc
import tables
import torch
import argparse
import os
import numpy as np


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
    parser.add_argument('--num_ite', default=5, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--epochs', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--mu', default=1.5, type=float, help='')
    parser.add_argument('--tau_ff', default=10, type=int, help='')
    parser.add_argument('--n_basis_ff', default=8, type=int, help='')
    parser.add_argument('--tau_fb', default=10, type=int, help='')
    parser.add_argument('--dropout_rate', default=None, type=float, help='')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--kappa', default=0.2, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha softmax coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay factor')
    parser.add_argument('--beta_2', default=0.999, type=float)
    parser.add_argument('--gamma', default=1., type=float, help='KL regularization factor')
    parser.add_argument('--r', default=0.3, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--suffix', type=str, default='', help='')
    parser.add_argument('--labels', nargs='+', default=None, type=int)
    parser.add_argument('--T', type=float, default=1., help='temperature')

    args = parser.parse_args()


distant_data_path = r'/users/k1804053/snn/multivalued_snn/'
local_data_path = r'C:/Users/K1804053/PycharmProjects/datasets/'
save_path = os.getcwd() + r'/results'

datasets = {'mnist_dvs_2': r'mnist_dvs_25ms_26pxl_2_digits.hdf5',
            'mnist_dvs_10': r'mnist_dvs_binary_25ms_26pxl_10_digits.hdf5',
            'mnist_dvs_10_polarity': r'mnist_dvs_binary_polarity_25ms_26pxl_10_digits.hdf5'
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
    if args.dataset[:5] == 'mnist':
        dataset = r'/home/k1804053/' + datasets[args.dataset]

train_shape = tables.open_file(dataset).root.train.data[:].shape
test_shape = tables.open_file(dataset).root.test.data[:].shape

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print(args.disable_cuda)

### Network parameters
n_inputs_enc = tables.open_file(dataset).root.stats.train_data[:][1]

n_outputs_dec = tables.open_file(dataset).root.train.label[:].shape[1]
n_hidden_enc = 32
n_outputs_enc = n_hidden_enc

n_inputs_dec = n_inputs_enc + n_outputs_enc
n_hidden_dec = 32

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

learning_rate = args.lr #/ (n_hidden_dec + n_hidden_enc)
kappa = args.kappa
alpha = args.alpha
beta = args.beta
beta_2 = args.beta_2
gamma = args.gamma
r = args.r
num_ite = args.num_ite

# Test parameters
ite_test = [50, 100, 200, 300, 400, 500, 800, 1000]

name = r'_' +  r'%d_epochs_nh_%d' % (num_samples_train, n_hidden_dec) + args.suffix
save_path = os.getcwd() + r'/results/' + args.dataset + name + '.pkl'
save_path_weights = None  #os.getcwd() + r'/results/' + args.dataset + name + '_weights.hdf5'

test_accs = {i: [] for i in ite_test}

dataset = tables.open_file(dataset)

### Find indices
if args.labels is not None:
    print(args.labels)
    indices = np.random.choice(misc.find_train_indices_for_labels(dataset, args.labels), [num_samples_train], replace=True)
    num_samples_test = min(num_samples_test, len(misc.find_test_indices_for_labels(dataset, args.labels)))
    test_indices = np.random.choice(misc.find_test_indices_for_labels(dataset, args.labels), [num_samples_test], replace=False)
else:
    indices = np.random.choice(np.arange(train_shape[0]), [num_samples_train], replace=True)
    test_indices = np.random.choice(np.arange(test_shape[0]), [num_samples_test], replace=False)






encoder = SNNetwork(**misc.make_network_parameters(n_inputs_enc, 0, n_hidden_enc), device=args.device)
decoder = SNNetwork(**misc.make_network_parameters(n_inputs_dec, n_outputs_dec, n_hidden_dec), device=args.device)





encoder.set_mode('train')
decoder.set_mode('train')

eligibility_trace_hidden_enc = {parameter: encoder.gradients[parameter]for parameter in encoder.gradients}
eligibility_trace_hidden_dec = {parameter: decoder.gradients[parameter][decoder.hidden_neurons - decoder.n_input_neurons] for parameter in decoder.gradients}
eligibility_trace_output_dec = {parameter: decoder.gradients[parameter][decoder.output_neurons - decoder.n_input_neurons] for parameter in decoder.gradients}

learning_signal = 0

baseline_num_enc = {parameter: eligibility_trace_hidden_enc[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden_enc}
baseline_den_enc = {parameter: eligibility_trace_hidden_enc[parameter].pow(2) for parameter in eligibility_trace_hidden_enc}

baseline_num_dec = {parameter: eligibility_trace_hidden_dec[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden_dec}
baseline_den_dec = {parameter: eligibility_trace_hidden_dec[parameter].pow(2) for parameter in eligibility_trace_hidden_dec}

S_prime = dataset.root.train.data.shape[-1]
noise_level = 5.






for j, sample_idx in enumerate(indices):
    if (j + 1) % dataset.root.train.data[:].shape[0] == 0:
        learning_rate /= 2

    if test_accs:
        if (j + 1) in test_accs:
            acc = utils_wispike.get_acc(encoder, decoder, dataset, test_indices, noise_level)
            print('test accuracy at ite %d: %f' % (int(j + 1), acc))

            encoder.set_mode('train')
            decoder.set_mode('train')

    misc.refractory_period(encoder)
    misc.refractory_period(decoder)

    sample_enc = torch.FloatTensor(dataset.root.train.data[sample_idx]).to(encoder.device)
    output_dec = torch.FloatTensor(dataset.root.train.label[sample_idx]).to(decoder.device)

    for s in range(S_prime):
        # Feedforward sampling encoder
        log_proba_enc = encoder(sample_enc[:, s])
        proba_hidden_enc = torch.sigmoid(encoder.potential[encoder.hidden_neurons - encoder.n_input_neurons])

        # pass the channel
        decoder_input = torch.where((utils_wispike.channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1])),
                                                           decoder.device, noise_level)) > 0.5, torch.tensor(1.), torch.tensor(0.))
        sample_dec = torch.cat((decoder_input, output_dec[:, s]), dim=0).to(decoder.device)
        # print(n_outputs_enc)

        log_proba_dec = decoder(sample_dec)
        proba_hidden_dec = torch.sigmoid(decoder.potential[decoder.hidden_neurons - decoder.n_input_neurons])

        ls = torch.sum(log_proba_dec[decoder.output_neurons - decoder.n_input_neurons]) \
             - alpha * torch.sum(torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1]))
                                 * torch.log(1e-12 + torch.cat((proba_hidden_enc, proba_hidden_dec)) / r)
                                 + (1 - torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1])))
                                    * torch.log(1e-12 + (1. - torch.cat((proba_hidden_enc, proba_hidden_dec))) / (1 - r)))

        # Local feedback and update
        learning_signal = kappa * learning_signal + (1 - kappa) * ls

        # Update parameters
        for parameter in encoder.gradients:
            eligibility_trace_hidden_enc[parameter].mul_(kappa).add_(1 - kappa, encoder.gradients[parameter])

            baseline_num_enc[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden_enc[parameter].pow(2).mul_(learning_signal))
            baseline_den_enc[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden_enc[parameter].pow(2))
            baseline = (baseline_num_enc[parameter]) / (baseline_den_enc[parameter] + 1e-07)

            encoder.get_parameters()[parameter] += learning_rate * (learning_signal - baseline) * eligibility_trace_hidden_enc[parameter]

        for parameter in decoder.gradients:
            eligibility_trace_hidden_dec[parameter].mul_(kappa).add_(1 - kappa, decoder.gradients[parameter][decoder.hidden_neurons - decoder.n_input_neurons])
            eligibility_trace_output_dec[parameter].mul_(kappa).add_(1 - kappa, decoder.gradients[parameter][decoder.output_neurons - decoder.n_input_neurons])

            baseline_num_dec[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden_dec[parameter].pow(2).mul_(learning_signal))
            baseline_den_dec[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden_dec[parameter].pow(2))
            baseline = (baseline_num_dec[parameter]) / (baseline_den_dec[parameter] + 1e-07)

            decoder.get_parameters()[parameter][decoder.hidden_neurons - decoder.n_input_neurons] \
                += learning_rate * (learning_signal - baseline) * eligibility_trace_hidden_dec[parameter]
            decoder.get_parameters()[parameter][decoder.output_neurons - decoder.n_input_neurons] += learning_rate * eligibility_trace_output_dec[parameter]

    if j % max(1, int(len(indices) / 5)) == 0:
        print('Step %d out of %d' % (j, len(indices)))

