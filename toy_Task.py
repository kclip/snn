from binary_snn.models.SNN import SNNetwork
from binary_snn.utils_binary import misc
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

save_path = os.getcwd() + r'/results'

args.disable_cuda = str2bool(args.disable_cuda)
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print(args.disable_cuda)

dataset = torch.bernoulli(torch.ones([1000, 8, 80]) * 0.5)

### Network parameters
n_inputs = dataset.shape[1]

n_outputs = n_inputs
n_hidden = n_inputs

### Learning parameters
if args.num_samples:
    num_samples_train = args.num_samples
else:
    num_samples_train = dataset.shape[0]

if args.num_samples_test:
    num_samples_test = args.num_samples_test
else:
    num_samples_test = dataset.shape[0]

learning_rate = args.lr
kappa = args.kappa
alpha = args.alpha
beta = args.beta
beta_2 = args.beta_2
gamma = args.gamma
r = args.r

# Test parameters
# ite_test = [50, 100, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
ite_test = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

save_path = None
save_path_weights = None 


### Find indices
indices = np.random.choice(np.arange(dataset.shape[0]), [num_samples_train], replace=True)
test_indices = np.random.choice(np.arange(dataset.shape[0]), [num_samples_test], replace=False)


def get_acc(network, dataset, test_indices, n_layers):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.set_mode('test')
    network.reset_internal_state()

    S_prime = dataset.shape[-1]
    outputs = torch.zeros([len(test_indices), network.n_output_neurons, S_prime])

    hidden_hist = torch.zeros([network.n_hidden_neurons, S_prime])

    for j, sample_idx in enumerate(test_indices):
        misc.refractory_period(network)
        sample = dataset[sample_idx].to(network.device)

        for s in range(S_prime):
            _ = network(sample[:, s])
            outputs[j, :, s] = network.spiking_history[network.output_neurons, -1]

            hidden_hist[:, s] = network.spiking_history[network.hidden_neurons, -1]

    # acc = float(torch.sum(outputs[:, :, 2:] == dataset[test_indices, :, :-2])) / dataset[test_indices, :, :-2].numel()
    acc = float(torch.sum(outputs[:, :, (1 + n_layers):] == dataset[test_indices, :, :-(1 + n_layers)])) / dataset[test_indices, :, :-(1 + n_layers)].numel()

    return acc


topology = torch.ones([n_hidden + n_outputs, n_inputs + n_hidden + n_outputs])
topology[-n_outputs:, :n_inputs] = 0
topology[:n_inputs, -n_outputs:] = 0

topology = torch.zeros([n_hidden + n_outputs, n_inputs + n_hidden + n_outputs])
topology[[i for i in range(n_hidden + n_outputs)], [i for i in range(n_hidden + n_outputs)]] = 1
topology[-n_outputs:, n_inputs:-n_outputs] = 1
# topology[-n_outputs:, n_inputs:] = 1
topology[:n_hidden, :n_inputs] = 1
topology[:n_hidden, n_inputs:(n_inputs + n_hidden)] = 1
n_layers = 1

print(topology)

# n_layers = 2
# n_neurons_per_layer = n_hidden // n_layers
#
# topology = torch.zeros([(n_hidden + n_outputs), (n_inputs + n_hidden + n_outputs)])
#
# topology[:n_neurons_per_layer, :n_inputs] = 1
#
# for i in range(1, n_layers + 1):
#     topology[i * n_neurons_per_layer: (i + 1) * n_neurons_per_layer, n_inputs + (i - 1) * n_neurons_per_layer: n_inputs + i * n_neurons_per_layer] = 1
#
# topology[:n_hidden, -n_outputs:] = 0
#
# topology[-n_outputs:, n_inputs:-n_outputs] = 1


network = SNNetwork(**misc.make_network_parameters(n_inputs, n_outputs, n_hidden, topology_type='custom', topology=topology, initialization='glorot', weights_magnitude=2.), device=args.device)
# network.import_weights(os.getcwd() + r'/results/toy_task_weights.hdf5')
print(torch.mean(torch.abs(network.get_parameters()['ff_weights']), dim=-1))

network.set_mode('train')

eligibility_trace_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.gradients}
eligibility_trace_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.gradients}

learning_signal = 0

baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden}
baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

S_prime = dataset.shape[-1]


for j, sample_idx in enumerate(indices):
    if (j + 1) % dataset.shape[0] == 0:
        learning_rate /= 2

    if (j + 1) in ite_test:
        acc = get_acc(network, dataset, test_indices, n_layers)
        print('test accuracy at ite %d: %f' % (int(j + 1), acc))

        network.set_mode('train')

    misc.refractory_period(network)

    # sample = torch.cat((dataset[sample_idx], torch.cat((torch.zeros([dataset.shape[1], 2]), dataset[sample_idx, :, :-2]), dim=-1))).to(network.device)
    sample = torch.cat((dataset[sample_idx], torch.cat((torch.zeros([dataset.shape[1], 1 + n_layers]), dataset[sample_idx, :, :-(1 + n_layers)]), dim=-1))).to(network.device)

    for s in range(S_prime):
        # Feedforward sampling encoder
        log_proba = network(sample[:, s])

        proba_hidden = torch.sigmoid(network.potential[network.hidden_neurons - network.n_input_neurons])

        ls_tmp = torch.sum(log_proba[network.output_neurons - network.n_input_neurons]) \
             - alpha * torch.sum(network.spiking_history[network.hidden_neurons, -1]* torch.log(1e-12 + proba_hidden / r)
                                 + (1 - network.spiking_history[network.hidden_neurons, -1]) * torch.log(1e-12 + (1. - proba_hidden) / (1 - r)))

        # Local feedback and update
        # print(network.potential[network.output_neurons - network.n_input_neurons])
        #
        # print(network.spiking_history[network.input_neurons, -2])
        # print(network.spiking_history[network.output_neurons, -1])
        # print(log_proba)
        # print('////')

        learning_signal = kappa * learning_signal + (1 - kappa) * ls_tmp

        # Update parameters
        for parameter in network.gradients:
            eligibility_trace_hidden[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.hidden_neurons - network.n_input_neurons])
            eligibility_trace_output[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.output_neurons - network.n_input_neurons])

            baseline_num[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2).mul_(learning_signal))
            baseline_den[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2))
            baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

            network.get_parameters()[parameter][network.hidden_neurons - network.n_input_neurons] \
                += learning_rate * (learning_signal - baseline) * eligibility_trace_hidden[parameter]


            network.get_parameters()[parameter][network.output_neurons - network.n_input_neurons] += learning_rate * eligibility_trace_output[parameter]


    if j % max(1, int(len(indices) / 5)) == 0:
        print('Step %d out of %d' % (j, len(indices)))

# print(torch.sum(network.get_parameters()['ff_weights'], dim=-1))
# print(network.get_parameters()['ff_weights'])
# print(network.get_parameters()['fb_weights'])
# print(network.get_parameters()['bias'])
