from models.SNN import SNNetwork
import pickle
from data_preprocessing import misc
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
    parser.add_argument('--weights', type=str, default=None, help='Path to weights to load')
    parser.add_argument('--mode', default='train_ml_online', help='Feedforward or interactive readout')
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
    parser.add_argument('--r', default=0.5, type=float, help='Desired spiking sparsity of the hidden neurons')
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--suffix', type=str, default='', help='')
    parser.add_argument('--labels', nargs='+', default=None, type=int)
    parser.add_argument('--T', type=float, default=1., help='temperature')

    args = parser.parse_args()

save_path = os.getcwd() + r'/results'

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print(args.disable_cuda)

dataset = torch.bernoulli(torch.ones([1000, 4, 80]) * 0.5)

### Network parameters
n_inputs_enc = dataset.shape[1]
n_outputs_enc = 0
n_hidden_enc = 0 + n_inputs_enc

n_inputs_dec = n_inputs_enc
n_outputs_dec = n_inputs_enc
n_hidden_dec = 0

### Learning parameters
if args.num_samples:
    num_samples_train = args.num_samples
else:
    num_samples_train = dataset.shape[0]

if args.num_samples_test:
    num_samples_test = args.num_samples_test
else:
    num_samples_test = dataset.shape[0]

learning_rate = args.lr / (n_hidden_dec + n_hidden_enc)
kappa = args.kappa
alpha = args.alpha
beta = args.beta
beta_2 = args.beta_2
gamma = args.gamma
r = args.r
num_ite = args.num_ite

# Test parameters
ite_test = [50, 100, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000, 7500, 10000]

name = r'_%d_epochs_nh_%d' % (num_samples_train, n_hidden_dec) + args.suffix
save_path = os.getcwd() + r'/results/' + 'toy_example' + name + '.pkl'
save_path_weights = None  #os.getcwd() + r'/results/' + args.dataset + name + '_weights.hdf5'

if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
        test_accs = pickle.load(f)
else:
    test_accs = {i: [] for i in ite_test}


### Find indices
indices = np.random.choice(np.arange(dataset.shape[0]), [num_samples_train], replace=True)
test_indices = np.random.choice(np.arange(dataset.shape[0]), [num_samples_test], replace=False)


def channel(channel_input, device, noise_level):
    channel_output = channel_input + torch.normal(0., torch.ones(channel_input.shape) * noise_level).to(device)
    return channel_output.round()


def get_acc(encoder, decoder, dataset, test_indices, noise_level):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoder.set_mode('test')
    encoder.reset_internal_state()

    decoder.set_mode('test')
    decoder.reset_internal_state()

    S_prime = dataset.shape[-1]
    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, S_prime])

    hidden_hist = torch.zeros([encoder.n_hidden_neurons + decoder.n_hidden_neurons, S_prime])

    for j, sample_idx in enumerate(test_indices):
        misc.refractory_period(encoder)
        misc.refractory_period(decoder)
        sample_enc = dataset[sample_idx].to(encoder.device)

        for s in range(S_prime):
            _ = encoder(sample_enc[:, s])
            decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1], decoder.device, noise_level).to(decoder.device)

            _ = decoder(decoder_input)
            outputs[j, :, s] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, s] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, s] = decoder.spiking_history[decoder.hidden_neurons, -1]


        # print(torch.sum(hidden_hist[:encoder.n_hidden_neurons], dim=-1))
        # print(torch.sum(hidden_hist[encoder.n_hidden_neurons:], dim=-1))
        print(outputs[j, :, :5])
        print(dataset[sample_idx, :, :5])
        print('//////////////////////////////////////////////////////////')

    acc = float(torch.sum(outputs[:, :, 1:] == dataset[test_indices, :, :-1])) / dataset[test_indices].numel()

    return acc


encoder = SNNetwork(**misc.make_network_parameters(n_inputs_enc, 0, n_hidden_enc), device=args.device)
decoder = SNNetwork(**misc.make_network_parameters(n_inputs_dec, n_outputs_dec, n_hidden_dec, topology_type='fully_connected'), device=args.device)

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

S_prime = dataset.shape[-1]
noise_level = 0.


for j, sample_idx in enumerate(indices):
    if (j + 1) % dataset.shape[0] == 0:
        learning_rate /= 2

    if test_accs:
        if (j + 1) in test_accs:
            acc = get_acc(encoder, decoder, dataset, test_indices, noise_level)
            test_accs[int(j + 1)].append(acc)
            print('test accuracy at ite %d: %f' % (int(j + 1), acc))

            encoder.set_mode('train')
            decoder.set_mode('train')

    misc.refractory_period(encoder)
    misc.refractory_period(decoder)

    sample_enc = dataset[sample_idx].to(encoder.device)
    output_dec = torch.cat((torch.zeros([dataset.shape[1], 1]), dataset[sample_idx, :, :-1]), dim=-1).to(decoder.device)

    for s in range(S_prime):
        # Feedforward sampling encoder
        log_proba_enc = encoder(sample_enc[:, s])
        proba_hidden_enc = torch.sigmoid(encoder.potential[encoder.hidden_neurons - encoder.n_input_neurons])

        # pass the channel
        # decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons, -1], decoder.device, noise_level).to(decoder.device)
        decoder_input = encoder.spiking_history[encoder.hidden_neurons, -1]

        sample_dec = torch.cat((decoder_input, output_dec[:, s]), dim=0)

        log_proba_dec = decoder(sample_dec)
        proba_hidden_dec = torch.sigmoid(decoder.potential[decoder.hidden_neurons - decoder.n_input_neurons])

        # ls_tmp = torch.sum(log_proba_dec[decoder.output_neurons - decoder.n_input_neurons]) \
        #      - alpha * torch.sum(torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1]))
        #                          * torch.log(1e-12 + torch.cat((proba_hidden_enc, proba_hidden_dec)) / r)
        #                          + (1 - torch.cat((encoder.spiking_history[encoder.hidden_neurons, -1], decoder.spiking_history[decoder.hidden_neurons, -1])))
        #                             * torch.log(1e-12 + (1. - torch.cat((proba_hidden_enc, proba_hidden_dec))) / (1 - r)))

        ls_tmp = torch.sum(log_proba_dec[decoder.output_neurons - decoder.n_input_neurons]) \
             - alpha * torch.sum(encoder.spiking_history[encoder.hidden_neurons, -1]* torch.log(1e-12 + proba_hidden_enc / r)
                                 + (1 - encoder.spiking_history[encoder.hidden_neurons, -1]) * torch.log(1e-12 + (1. - proba_hidden_enc) / (1 - r)))

        # Local feedback and update
        learning_signal = kappa * learning_signal + (1 - kappa) * ls_tmp

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

    # print(encoder.spiking_history[:n_inputs_enc])
    # print(decoder.spiking_history[-n_inputs_enc:])

    if j % max(1, int(len(indices) / 5)) == 0:
        print('Step %d out of %d' % (j, len(indices)))

