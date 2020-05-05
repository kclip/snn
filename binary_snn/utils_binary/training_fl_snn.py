import torch
import tables
import pickle
from binary_snn.utils_binary.misc import refractory_period, get_acc_and_loss


def feedforward_sampling(network, example, ls, et, alpha, r, gradients_accum=None):
    log_proba = network(example)

    # Accumulate learning signal
    proba_hidden = torch.sigmoid(network.potential[network.hidden_neurons - network.n_input_neurons])
    ls += torch.sum(log_proba[network.output_neurons - network.n_input_neurons]) \
          - alpha*torch.sum(network.spiking_history[network.hidden_neurons, -1]
          * torch.log
                            (1e-12 + proba_hidden / r)
          + (1 - network.spiking_history[network.hidden_neurons, -1]) * torch.log(1e-12 + (1. - proba_hidden) / (1 - r)))

    for parameter in network.gradients:
        # Only when the comm. rate is fixed
        if (parameter == 'ff_weights') & (gradients_accum is not None):
            gradients_accum += torch.abs(network.get_gradients()[parameter])

        et[parameter] += network.gradients[parameter]

    return log_proba, ls, et, gradients_accum


def local_feedback_and_update(network, eligibility_trace, et_temp, learning_signal, ls_temp, learning_rate, beta, kappa, s, deltas):
    """"
    Runs the local feedback and update steps:
    - computes the learning signal
    - updates the learning parameter
    """
    # At local algorithmic timesteps, do a local update
    if (s + 1) % deltas == 0:
        # local feedback
        learning_signal = kappa * learning_signal + (1 - kappa) * ls_temp
        ls_temp = 0

        # Update parameter
        for parameter in network.gradients:
            eligibility_trace[parameter].mul_(kappa).add_(1 - kappa, et_temp[parameter])
            et_temp[parameter] = 0

            network.get_parameters()[parameter] += learning_rate * eligibility_trace[parameter]

    return eligibility_trace, et_temp, learning_signal, ls_temp


def train(network, dataset, indices, test_indices, test_accs, learning_rate, alpha, beta, kappa, deltas, r, start_idx, save_path=None):
    """"
    Train a network on the sequence passed as argument.
    """

    network.set_mode('train')

    eligibility_trace_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.gradients}
    eligibility_trace_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.gradients}

    et_temp_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.gradients}
    et_temp_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.gradients}

    learning_signal = 0
    ls_temp = 0

    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

    S_prime = tables.open_file(dataset).root.train.label[:].shape[-1]
    S = len(indices[start_idx:]) * S_prime

    dataset = tables.open_file(dataset)

    for s in range(S):
        if s % S_prime == 0:
            refractory_period(network)

            if (int(s / S_prime) + 1) % dataset.root.train.data[:].shape[0] == 0:
                learning_rate /= 2

            if test_accs:
                if (int(s / S_prime) + 1) in test_accs:
                    acc, loss = get_acc_and_loss(network, dataset, test_indices)
                    test_accs[int(s / S_prime + 1)].append(acc)
                    print('test accuracy at ite %d: %f' % (int(s / S_prime + 1), acc))

                    if save_path is not None:
                        with open(save_path, 'wb') as f:
                            pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)

                    network.set_mode('train')

            sample = torch.cat((torch.FloatTensor(dataset.root.train.data[indices[int(s / S_prime)]]),
                                torch.FloatTensor(dataset.root.train.label[indices[int(s / S_prime)]])), dim=0)

        # Feedforward sampling step
        log_proba, ls_temp, et_temp_hidden, et_temp_output = feedforward_sampling(network, sample[:, s % S_prime], ls_temp, et_temp_hidden, et_temp_output, alpha, r)

        # Local feedback and update
        eligibility_trace_hidden, eligibility_trace_output, et_temp_hidden, et_temp_output, learning_signal, ls_temp, baseline_num, baseline_den \
            = local_feedback_and_update(network, eligibility_trace_hidden, eligibility_trace_output, et_temp_hidden, et_temp_output,
                                        learning_signal, ls_temp, baseline_num, baseline_den, learning_rate, beta, kappa, s, deltas)

        if s % int(S / 5) == 0:
            print('Step %d out of %d' % (s, S))

    return test_accs
