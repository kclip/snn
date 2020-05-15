import torch
import pickle
from binary_snn.utils_binary.misc import refractory_period, get_acc_and_loss, get_train_acc_and_loss


def feedforward_sampling(network, example, alpha, r):
    """"
    Feedforward sampling step:
    - passes information through the network
    - computes the learning signal
    """

    log_proba = network(example)

    # Accumulate learning signal
    proba_hidden = torch.sigmoid(network.potential[network.hidden_neurons - network.n_input_neurons])
    ls = torch.sum(log_proba[network.output_neurons - network.n_input_neurons]) \
          - alpha*torch.sum(network.spiking_history[network.hidden_neurons, -1]
          * torch.log(1e-12 + proba_hidden / r)
          + (1 - network.spiking_history[network.hidden_neurons, -1]) * torch.log(1e-12 + (1. - proba_hidden) / (1 - r)))

    return log_proba, ls


def local_feedback_and_update(network, ls_tmp, eligibility_trace_hidden, eligibility_trace_output,
                              learning_signal, baseline_num, baseline_den, learning_rate, beta, kappa):
    """"
    Runs the local feedback and update steps:
    - computes the learning signal
    - updates the learning parameter
    """

    # local feedback
    learning_signal = kappa * learning_signal + (1 - kappa) * ls_tmp

    # Update parameter
    for parameter in network.gradients:
        eligibility_trace_hidden[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.hidden_neurons - network.n_input_neurons])

        baseline_num[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2).mul_(learning_signal))
        baseline_den[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2))
        baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        network.get_parameters()[parameter][network.hidden_neurons - network.n_input_neurons] \
            += learning_rate * (learning_signal - baseline) * eligibility_trace_hidden[parameter]

        if eligibility_trace_output is not None:
            eligibility_trace_output[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.output_neurons - network.n_input_neurons])
            network.get_parameters()[parameter][network.output_neurons - network.n_input_neurons] += learning_rate * eligibility_trace_output[parameter]

    return eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den


def train(network, dataset, indices, test_indices, test_accs, learning_rate, alpha, beta, kappa, r, start_idx, args, save_path=None):
    """"
    Train a network.
    """

    network.set_mode('train')

    eligibility_trace_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.gradients}
    eligibility_trace_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.gradients}

    learning_signal = 0

    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

    S_prime = dataset.root.stats.train_label[:][-1]

    for j, sample_idx in enumerate(indices[start_idx:]):
        j += start_idx
        if (j + 1) % 5 * (dataset.root.train.data[:].shape[0]) == 0:
            learning_rate /= 2

        # Regularly test the accuracy
        if test_accs:
            if (j + 1) in test_accs:
                acc, loss = get_acc_and_loss(network, dataset, test_indices)
                test_accs[int(j + 1)].append(acc)
                print('test accuracy at ite %d: %f' % (int(j + 1), acc))

                # acc_train, _ = get_train_acc_and_loss(network, dataset, args.labels)
                # print('train accuracy at ite %d: %f' % (int(j + 1), acc_train))

                if save_path is not None:
                    with open(save_path, 'wb') as f:
                        pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)

                    network.set_mode('train')

        refractory_period(network)
        sample = torch.cat((torch.FloatTensor(dataset.root.train.data[sample_idx]),
                            torch.FloatTensor(dataset.root.train.label[sample_idx])), dim=0).to(network.device)
        for s in range(S_prime):
            # Feedforward sampling
            log_proba, ls_tmp = feedforward_sampling(network, sample[:, s], alpha, r)
            # Local feedback and update
            eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den \
                = local_feedback_and_update(network, ls_tmp, eligibility_trace_hidden, eligibility_trace_output,
                                            learning_signal, baseline_num, baseline_den, learning_rate, beta, kappa)

        if j % max(1, int(len(indices) / 5)) == 0:
            print('Step %d out of %d' % (j, len(indices)))

    return test_accs
