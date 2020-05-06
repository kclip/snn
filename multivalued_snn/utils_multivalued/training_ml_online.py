import torch
import tables
from multivalued_snn.utils_multivalued.misc import refractory_period, get_acc_and_loss, get_train_acc_and_loss
import pickle


def feedforward_sampling_ml(network, training_sequence, r, gamma):
    log_proba = network(training_sequence)

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
        eligibility_trace_hidden[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.hidden_neurons - network.n_input_neurons])

        baseline_num[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2).mul_(reward))
        baseline_den[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2))
        baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        updates_hidden[parameter].mul_(kappa).add_(1 - kappa, (reward - baseline) * eligibility_trace_hidden[parameter])

        network.get_parameters()[parameter][network.hidden_neurons - network.n_input_neurons] += (learning_rate * updates_hidden[parameter])

        if eligibility_trace_output is not None:
            eligibility_trace_output[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.output_neurons - network.n_input_neurons])
            network.get_parameters()[parameter][network.output_neurons - network.n_input_neurons] += (learning_rate * eligibility_trace_output[parameter])

    return baseline_num, baseline_den, updates_hidden, eligibility_trace_hidden, eligibility_trace_output


def train(network, dataset, indices, test_indices, test_accs, learning_rate, kappa, beta, gamma, r, start_idx=0, save_path=None, save_path_weights=None):
    """"
    Train a network on the sequence passed as argument.
    """

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_input_neurons, :, -network.n_output_neurons:]) == 0,\
        'There must be no backward connection from output to hidden neurons.'
    network.set_mode('train_ml')

    num_samples_train = dataset.root.stats.train[:][0]
    S_prime = dataset.root.stats.train[:][-1]

    eligibility_trace_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.gradients}
    eligibility_trace_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.gradients}
    updates_hidden = {parameter: eligibility_trace_hidden[parameter] for parameter in network.gradients}

    reward = 0
    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2)*reward for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

    for j, sample_idx in enumerate(indices[start_idx:]):
        j += start_idx
        if test_accs:
            if (j + 1) in test_accs:
                acc, _ = get_acc_and_loss(network, dataset, test_indices)
                test_accs[int(j + 1)].append(acc)

                print('test accuracy at ite %d: %f' % (int(j + 1), acc))

                if save_path is not None:
                    with open(save_path, 'wb') as f:
                        pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)
                if save_path_weights is not None:
                    network.save(save_path_weights)

                network.set_mode('train_ml')

        refractory_period(network)

        sample = torch.cat((torch.FloatTensor(dataset.root.train.data[sample_idx]),
                            torch.FloatTensor(dataset.root.train.label[sample_idx])), dim=0)

        for s in range(S_prime):
            reward = feedforward_sampling_ml(network, sample[:, :, s].to(network.device), r, gamma)
            baseline_num, baseline_den, updates_hidden, eligibility_trace_hidden, eligibility_trace_output = \
                ml_update(network, eligibility_trace_hidden, eligibility_trace_output, reward, updates_hidden, baseline_num, baseline_den, learning_rate, kappa, beta)

        if (j + 1) % (3 * num_samples_train) == 0:
            learning_rate /= 2

        if j % max(1, int(len(indices) / 5)) == 0:
            print('Sample %d out of %d' % (j + 1, len(indices)))

    dataset.close()
    try:
        return test_accs
    except:
        return
