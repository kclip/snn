import os
import torch

from snn.utils.utils_snn import refractory_period, test


def feedforward_sampling(network, inputs, outputs, gamma, r):
    """"
    Feedforward sampling step:
    - passes information through the network
    - computes the learning signal
    """

    log_proba = network(inputs, outputs)

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
        eligibility_trace_hidden[parameter].mul_(kappa).add_(network.get_gradients()[parameter][network.hidden_neurons - network.n_input_neurons], alpha=1 - kappa)

        baseline_num[parameter].mul_(beta).add_(eligibility_trace_hidden[parameter].pow(2).mul(learning_signal), alpha=1 - beta)
        baseline_den[parameter].mul_(beta).add_(eligibility_trace_hidden[parameter].pow(2), alpha=1 - beta)
        baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        network.get_parameters()[parameter][network.hidden_neurons - network.n_input_neurons] \
            += lr * (learning_signal - baseline) * eligibility_trace_hidden[parameter]

        if eligibility_trace_output is not None:
            eligibility_trace_output[parameter].mul_(kappa).add_(network.get_gradients()[parameter][network.output_neurons - network.n_input_neurons], alpha=1 - kappa)
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


def train_on_example(network, T, inputs, outputs, gamma, r, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den, lr, beta, kappa):
    for t in range(T):
        # Feedforward sampling
        log_proba, ls_tmp = feedforward_sampling(network, inputs[t], outputs[:, t], gamma, r)
        # Local feedback and update
        eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den \
            = local_feedback_and_update(network, ls_tmp, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den, lr, beta, kappa)

    return log_proba, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den


def train_experiment(network, args, params, train_dl, test_dl, train_accs, train_losses, test_accs, test_losses):
    """"
    Train an SNN.
    """

    eligibility_trace_output, eligibility_trace_hidden, \
        learning_signal, baseline_num, baseline_den = init_training(network)

    T_train = int(params['sample_length_train'] / params['dt'])
    T_test = int(params['sample_length_test'] / params['dt'])
    lr = params['lr']

    train_iterator = iter(train_dl)

    for j in range(params['n_examples_train'] - params['start_idx']):
        j += params['start_idx']

        # Regularly test the accuracy
        train_accs, train_losses, test_accs, test_losses = test(network, params, j, train_dl, T_train, test_dl, T_test,
                                                                params['test_period'], train_accs, train_losses, test_accs, test_losses, args.save_path)

        refractory_period(network)

        try:
            inputs, outputs = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dl)
            inputs, outputs = next(train_iterator)

        inputs = inputs[0].to(network.device)
        outputs = outputs[0].to(network.device)

        log_proba, eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den = \
            train_on_example(network, T_train, inputs, outputs, params['gamma'], params['r'], eligibility_trace_hidden,
                             eligibility_trace_output, learning_signal, baseline_num, baseline_den, lr, params['beta'], params['kappa'])

        if j % max(1, int(params['n_examples_train'] / 5)) == 0:
            print('Step %d out of %d' % (j, params['n_examples_train']))

    # At the end of training, save final weights if none exist or if this ite was better than all the others
    if not os.path.exists(args.save_path + '/network_weights_final.hdf5'):
        network.save(args.save_path + '/network_weights_final.hdf5')
    else:
        if test_accs[list(test_accs.keys())[-1]][-1] >= max(test_accs[list(test_accs.keys())[-1]][:-1]):
            network.save(args.save_path + '/network_weights_final.hdf5')

    return train_accs, train_losses, test_accs, test_losses
