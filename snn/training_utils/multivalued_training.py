import pickle
import os

import torch

from snn.utils.utils_wtasnn import refractory_period, test
from snn.data_preprocessing.load_data import get_example


def feedforward_sampling_ml(network, inputs, outputs, r, gamma):
    log_proba = network(inputs, outputs)

    probas = torch.softmax(torch.cat((torch.zeros([network.n_hidden_neurons, 1]).to(network.device),
                                      network.potential[network.hidden_neurons - network.n_input_neurons]), dim=-1), dim=-1)

    reg = \
        torch.sum(torch.sum(network.spiking_history[network.hidden_neurons, :, -1]
                            * torch.log(1e-07 + probas[:, 1:] / ((1 - r) / network.alphabet_size)), dim=-1)
                  + (1 - torch.sum(network.spiking_history[network.hidden_neurons, :, -1], dim=-1))
                  * torch.log(1e-07 + probas[:, 0] / r))

    reward = torch.sum(log_proba[network.output_neurons - network.n_input_neurons]) - gamma * reg

    return reward


def ml_update(network, eligibility_trace_hidden, eligibility_trace_output, reward,
              updates_hidden, baseline_num, baseline_den, learning_rate, kappa, beta):

    for parameter in updates_hidden:
        eligibility_trace_hidden[parameter].mul_(kappa).add_(network.get_gradients()[parameter][network.hidden_neurons - network.n_input_neurons], alpha=1 - kappa)

        baseline_num[parameter].mul_(beta).add_(eligibility_trace_hidden[parameter].pow(2).mul(reward), alpha=1 - beta)
        baseline_den[parameter].mul_(beta).add_(eligibility_trace_hidden[parameter].pow(2), alpha=1 - beta)
        baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        updates_hidden[parameter].mul_(kappa).add_((reward - baseline) * eligibility_trace_hidden[parameter], alpha=1 - kappa)

        network.get_parameters()[parameter][network.hidden_neurons - network.n_input_neurons] += (learning_rate * updates_hidden[parameter])

        if eligibility_trace_output is not None:
            eligibility_trace_output[parameter].mul_(kappa).add_(network.get_gradients()[parameter][network.output_neurons - network.n_input_neurons], alpha=1 - kappa)
            network.get_parameters()[parameter][network.output_neurons - network.n_input_neurons] += (learning_rate * eligibility_trace_output[parameter])

    return baseline_num, baseline_den, updates_hidden, eligibility_trace_hidden, eligibility_trace_output


def init_training(network):
    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_input_neurons, :, -network.n_output_neurons:]) == 0,\
        'There must be no backward connection from output to hidden neurons.'
    network.train()


    eligibility_trace_hidden = {parameter: network.get_gradients()[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.get_gradients()}
    eligibility_trace_output = {parameter: network.get_gradients()[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.get_gradients()}
    updates_hidden = {parameter: eligibility_trace_hidden[parameter] for parameter in network.get_gradients()}

    reward = 0
    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2)*reward for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}


    return eligibility_trace_output, eligibility_trace_hidden, updates_hidden, baseline_num, baseline_den, reward


def train_experiment(network, args, params, train_dl, test_dl, train_accs, train_losses, test_accs, test_losses):

    eligibility_trace_output, eligibility_trace_hidden, updates_hidden, baseline_num, baseline_den, reward = init_training(network)

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

        inputs = inputs[0].transpose(1, 2).to(network.device)
        outputs = outputs[0].to(network.device)
        print(inputs.shape, outputs.shape)
        for t in range(T_train):
            reward = feedforward_sampling_ml(network, inputs[t], outputs[:, :, t], params['r'], params['gamma'])
            baseline_num, baseline_den, updates_hidden, eligibility_trace_hidden, eligibility_trace_output = \
                ml_update(network, eligibility_trace_hidden, eligibility_trace_output, reward, updates_hidden, baseline_num, baseline_den, lr, params['kappa'], params['beta'])

        if j % max(1, int(params['n_examples_train'] / 5)) == 0:
            print('Sample %d out of %d' % (j + 1, params['n_examples_train']))

    # At the end of training, save final weights if none exist or if this ite was better than all the others
    if not os.path.exists(args.save_path + '/network_weights_final.hdf5'):
        network.save(args.save_path + '/network_weights_final.hdf5')
    else:
        if test_accs[list(test_accs.keys())[-1]][-1] >= max(test_accs[list(test_accs.keys())[-1]][:-1]):
            network.save(args.save_path + '/network_weights_final.hdf5')

    return train_accs, train_losses, test_accs, test_losses
