import os

import time
import torch
import pickle
from snn.utils.utils_snn import refractory_period, test
from snn.data_preprocessing.load_data import get_example


def feedforward_sampling(network, example, gamma, r):
    """"
    Feedforward sampling step:
    - passes information through the network
    - computes the learning signal
    """

    log_proba = network(example)

    # Accumulate learning signal
    proba_hidden = torch.sigmoid(network.potential[network.hidden_neurons - network.n_input_neurons])
    ls = torch.sum(log_proba[network.output_neurons - network.n_input_neurons]) \
          - gamma * torch.sum(network.spiking_history[network.hidden_neurons, -1]
          * torch.log(1e-07 + proba_hidden / r)
          + (1 - network.spiking_history[network.hidden_neurons, -1]) * torch.log(1e-07 + (1. - proba_hidden) / (1 - r)))

    return log_proba, ls


def local_feedback_and_update(network, ls_tmp, eligibility_trace_hidden, eligibility_trace_output,
                              learning_signal, baseline_num, baseline_den, lr, beta, kappa):
    """"
    Runs the local feedback and update steps:
    - computes the learning signal
    - updates the learning parameter
    """

    # local feedback
    if ls_tmp != 0:
        learning_signal = kappa * learning_signal + (1 - kappa) * ls_tmp

    # Update parameter
    for parameter in network.get_gradients():
        eligibility_trace_hidden[parameter].mul_(kappa).add_(1 - kappa, network.get_gradients()[parameter][network.hidden_neurons - network.n_input_neurons])

        baseline_num[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2).mul_(learning_signal))
        baseline_den[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2))
        baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        network.get_parameters()[parameter][network.hidden_neurons - network.n_input_neurons] \
            += lr * (learning_signal - baseline) * eligibility_trace_hidden[parameter]

        if eligibility_trace_output is not None:
            eligibility_trace_output[parameter].mul_(kappa).add_(1 - kappa, network.get_gradients()[parameter][network.output_neurons - network.n_input_neurons])
            network.get_parameters()[parameter][network.output_neurons - network.n_input_neurons] += lr * eligibility_trace_output[parameter]

    return eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den


def init_training(network):
    network.train()

    eligibility_trace_hidden = {parameter: network.get_gradients()[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.get_gradients()}
    eligibility_trace_output = {parameter: network.get_gradients()[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.get_gradients()}

    learning_signal = 0

    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

    return eligibility_trace_output, eligibility_trace_hidden, learning_signal, baseline_num, baseline_den


def train_on_example(network, T, example, gamma, r, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den, lr, beta, kappa):
    for t in range(T):
        # Feedforward sampling
        log_proba, ls_tmp = feedforward_sampling(network, example[:, t], gamma, r)
        # Local feedback and update
        eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den \
            = local_feedback_and_update(network, ls_tmp, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den, lr, beta, kappa)

    return log_proba, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den


def train_experiment(network, args, params):
    """"
    Train an SNN.
    """

    t0 = time.time()

    eligibility_trace_output, eligibility_trace_hidden, \
        learning_signal, baseline_num, baseline_den = init_training(network)

    train_data = params['dataset'].root.train
    test_data = params['dataset'].root.test
    T = int(params['sample_length'] * 1000 / params['dt'])
    x_max = params['dataset'].root.stats.train_data[1]
    lr = params['lr']

    for j, idx in enumerate(params['train_indices'][params['start_idx']:]):
        j += params['start_idx']

        if ((j + 1) % params['dataset'].root.stats.train_data[0]) == 0:
            lr /= 2

        # Regularly test the accuracy
        test(network, j, train_data, params['train_indices'], test_data, params['test_indices'], T, params['n_classes'], params['pattern'], params['input_shape'],
             params['dt'], x_max, params['polarity'], params['test_period'], params['train_accs'], params['train_losses'], params['test_accs'], params['test_losses'], args.save_path)

        refractory_period(network)

        inputs, label = get_example(train_data, idx, T, params['n_classes'], params['pattern'], params['input_shape'], params['dt'], x_max, params['polarity'])
        example = torch.cat((inputs, label), dim=0).to(network.device)

        log_proba, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den = \
            train_on_example(network, T, example, params['gamma'], params['r'], eligibility_trace_hidden,
                             eligibility_trace_output, learning_signal, baseline_num, baseline_den, lr, params['beta'], params['kappa'])

        if j % max(1, int(len(params['train_indices']) / 5)) == 0:
            print('Step %d out of %d' % (j, len(params['train_indices'])))

    # At the end of training, save final weights if none exist or if this ite was better than all the others
    if not os.path.exists(args.save_path + '/network_weights_final.hdf5'):
        network.save(args.save_path + '/network_weights_final.hdf5')
    else:
        if params['test_accs'][params['num_samples_train']][-1] >= max(params['test_accs'][params['num_samples_train']][:-1]):
            network.save(args.save_path + '/network_weights_final.hdf5')

    print(t0 - time.time())