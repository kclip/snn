import torch
import tables
import utils.filters as filters
import numpy as np
import pickle
from tqdm import tqdm


def custom_softmax(input_tensor, alpha, dim_):
    u = torch.max(input_tensor)
    return torch.exp(alpha * (input_tensor - u)) / (torch.exp(- alpha * u) + torch.sum(torch.exp(alpha * (input_tensor - u)), dim=dim_))[:, None]


def make_topology(mode, topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, n_neurons_per_layer=0, density=1):
    if topology_type == 'fully_connected':
        topology = torch.ones([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons], dtype=torch.float)

    elif topology_type == 'sparse':
        indices = np.random.choice(n_hidden_neurons * n_hidden_neurons, [int(density * n_hidden_neurons**2)], replace=False)

        row = np.array([int(index / n_hidden_neurons) for index in indices])
        col = np.array([int(index % n_hidden_neurons) for index in indices]) + n_input_neurons

        topology = torch.zeros([n_hidden_neurons + n_output_neurons, n_input_neurons + n_hidden_neurons + n_output_neurons])
        topology[[r for r in row], [c for c in col]] = 1
        topology[:, :n_input_neurons] = 1
        topology[-n_output_neurons:, :] = 1

    elif topology_type == 'layered':
        n_layers = n_hidden_neurons // n_neurons_per_layer
        assert (n_hidden_neurons % n_neurons_per_layer) == 0

        topology = torch.zeros([(n_hidden_neurons + n_output_neurons), (n_input_neurons + n_hidden_neurons + n_output_neurons)])

        topology[:, :n_input_neurons] = 1

        for i in range(1, n_layers + 1):
            topology[i * n_neurons_per_layer: (i + 1) * n_neurons_per_layer, n_input_neurons + (i - 1) * n_neurons_per_layer: n_input_neurons + i * n_neurons_per_layer] = 1

        topology[:n_hidden_neurons, -n_output_neurons:] = 0
        topology[-n_output_neurons:, -n_output_neurons:] = 1
        topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    if mode == 'train_ml':
        topology[:n_hidden_neurons, -n_output_neurons:] = 0
    elif mode == 'train_rl':
        topology[:n_hidden_neurons, -n_output_neurons:] = 1

    topology[-n_output_neurons:, -n_output_neurons:] = 1
    topology[[i for i in range(n_output_neurons + n_hidden_neurons)], [i + n_input_neurons for i in range(n_output_neurons + n_hidden_neurons)]] = 0

    assert torch.sum(topology[:, :n_input_neurons]) == (n_hidden_neurons + n_output_neurons) * n_input_neurons
    return topology


def make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode, topology_type='fully_connected', topology=None, n_neurons_per_layer=0,
                            density=1, gain_fb=-0.05, gain_ff=0.06, gain_bias=0.05,
                            initialization='uniform', connection_topology='full', n_basis_ff=8, ff_filter=filters.raised_cosine_pillow_08, n_basis_fb=1,
                            fb_filter=filters.raised_cosine_pillow_08, tau_ff=10, tau_fb=10, mu=1.5, task='supervised', dropout_rate=None):

    if topology_type != 'custom':
        topology = make_topology(mode, topology_type, n_input_neurons, n_output_neurons, n_hidden_neurons, n_neurons_per_layer, density)
    else:
        topology = topology

    print(topology[:, n_input_neurons:])
    network_parameters = {'n_input_neurons': n_input_neurons,
                          'n_output_neurons': n_output_neurons,
                          'n_hidden_neurons': n_hidden_neurons,
                          'topology': topology,
                          'alphabet_size': alphabet_size,
                          'n_basis_feedforward': n_basis_ff,
                          'feedforward_filter': ff_filter,
                          'n_basis_feedback': n_basis_fb,
                          'feedback_filter': fb_filter,
                          'tau_ff': tau_ff,
                          'tau_fb': tau_fb,
                          'mu': mu,
                          'initialization': initialization,
                          'connection_topology': connection_topology,
                          'gain_fb': gain_fb,
                          'gain_ff': gain_ff,
                          'gain_bias': gain_bias,
                          'task': task,
                          'mode': mode,
                          'dropout_rate': dropout_rate
                          }

    return network_parameters


def refractory_period(network):
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.visible_neurons), network.alphabet_size], dtype=torch.float).to(network.device))


def time_average(old, new, kappa):
    return old * kappa + new * (1 - kappa)


def get_acc_and_loss(network, dataset, test_indices):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.set_mode('test')
    network.reset_internal_state()

    S_prime = dataset.root.stats.test[:][-1]
    outputs = torch.zeros([len(test_indices), network.n_output_neurons, network.alphabet_size, S_prime])

    loss = 0
    # hidden_hist = torch.zeros([network.n_hidden_neurons, network.alphabet_size, S_prime])

    for j, sample_idx in enumerate(test_indices):
        refractory_period(network)

        sample = torch.FloatTensor(dataset.root.test.data[sample_idx])

        for s in range(S_prime):
            log_proba = network(sample[:, :, s].to(network.device))
            # loss += torch.sum(log_proba).numpy()
            outputs[j, :, :, s % S_prime] = network.spiking_history[network.output_neurons, :, -1]
            # hidden_hist[:, :, s] = network.spiking_history[network.hidden_neurons, :, -1]

        # print(torch.sum(hidden_hist, dim=(-1, -2)))

    predictions = torch.max(torch.sum(outputs, dim=(-1, -2)), dim=-1).indices
    true_classes = torch.max(torch.sum(torch.FloatTensor(dataset.root.test.label[:][test_indices]), dim=(-1, -2)), dim=-1).indices

    loss = 0
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss


def feedforward_sampling_ml(network, training_sequence, r, gamma):
    log_proba = network(training_sequence)

    probas = torch.softmax(torch.cat((torch.zeros([network.n_hidden_neurons, 1]).to(network.device), network.potential[network.hidden_neurons - network.n_non_learnable_neurons]), dim=-1), dim=-1)

    reg = torch.sum(torch.sum(network.spiking_history[network.hidden_neurons, :, -1] * torch.log(1e-07 + probas[:, 1:] / ((1 - r) / network.alphabet_size)), dim=-1)
                    + (1 - torch.sum(network.spiking_history[network.hidden_neurons, :, -1], dim=-1)) * torch.log(1e-07 + probas[:, 0] / r))

    reward = torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons]) - gamma * reg

    return reward


def ml_update(network, eligibility_trace_hidden, eligibility_trace_output, reward,
              updates_hidden, baseline_num, baseline_den, learning_rate, kappa, beta):

    for parameter in updates_hidden:
        eligibility_trace_hidden[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.hidden_neurons - network.n_non_learnable_neurons])
        eligibility_trace_output[parameter].mul_(kappa).add_(1 - kappa, network.gradients[parameter][network.output_neurons - network.n_non_learnable_neurons])

        baseline_num[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2).mul_(reward))
        baseline_den[parameter].mul_(beta).add_(1 - beta, eligibility_trace_hidden[parameter].pow(2))
        baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        updates_hidden[parameter].mul_(kappa).add_(1 - kappa, (reward - baseline) * eligibility_trace_hidden[parameter])

        network.get_parameters()[parameter][network.hidden_neurons - network.n_non_learnable_neurons] += (learning_rate * updates_hidden[parameter])
        network.get_parameters()[parameter][network.output_neurons - network.n_non_learnable_neurons] += (learning_rate * eligibility_trace_output[parameter])

    return baseline_num, baseline_den, updates_hidden, eligibility_trace_hidden, eligibility_trace_output


def train_ml_online(network, dataset, indices, test_indices, test_accs, learning_rate, kappa, beta, gamma, r, start_idx=0, save_path=None, save_path_weights=None):
    """"
    Train a network on the sequence passed as argument.
    """

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -network.n_output_neurons:]) == 0,\
        'There must be no backward connection from output to hidden neurons.'
    network.set_mode('train_ml')

    num_samples_train = tables.open_file(dataset).root.stats.train[:][0]
    S_prime = tables.open_file(dataset).root.stats.train[:][-1]

    eligibility_trace_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_non_learnable_neurons] for parameter in network.gradients}
    eligibility_trace_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_non_learnable_neurons] for parameter in network.gradients}
    updates_hidden = {parameter: eligibility_trace_hidden[parameter] for parameter in network.gradients}

    reward = 0
    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2)*reward for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

    dataset = tables.open_file(dataset)

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

        for s in (range(S_prime)):
            reward = feedforward_sampling_ml(network, sample[:, :, s].to(network.device), r, gamma)
            baseline_num, baseline_den, updates_hidden, eligibility_trace_hidden, eligibility_trace_output = \
                ml_update(network, eligibility_trace_hidden, eligibility_trace_output, reward, updates_hidden, baseline_num, baseline_den, learning_rate, kappa, beta)

        if (j + 1) % num_samples_train == 0:
            learning_rate /= 2

        if j % max(1, int(len(indices) / 5)) == 0:
            # print(torch.sum(network.feedforward_weights, dim=(-1, -2, -3, -4)))
            print('Sample %d out of %d' % (j + 1, len(indices)))

    dataset.close()
    try:
        return test_accs
    except:
        return
























































































































































def batch_update_ml(network, idx, rewards, baselines_num, baselines_den, spiking_history, potentials, learning_rate, S_prime, beta):
    assert len(rewards) == S_prime
    assert len(potentials) == S_prime

    accum_rewards = [0] * S_prime
    accum_rewards[-1] = rewards[-1]

    for i in range(S_prime - 2, -1, -1):
        accum_rewards[i] = accum_rewards[i + 1] + rewards[i]

    # Compute the baseline
    for s in range(S_prime):
        gradients = network.compute_gradients(spiking_history[network.learnable_neurons, :, s + 1], potentials[s],
                                              network.compute_ff_trace(spiking_history[:, :, max(0, s + 1 - network.memory_length): s + 1]),
                                              network.compute_fb_trace(spiking_history[:, :, max(0, s + 1 - network.memory_length): s + 1]))

        for parameter in gradients:
            if idx == 0:
                baselines_num[s][parameter] *= accum_rewards[s]
            else:
                baselines_num[s][parameter] = beta * baselines_num[s][parameter] + (1 - beta) * (gradients[parameter].pow(2) + 1e-12) * accum_rewards[s]
                baselines_den[s][parameter] = beta * baselines_den[s][parameter] + (1 - beta) * (gradients[parameter].pow(2) + 1e-12)

            baseline = baselines_num[s][parameter] / baselines_den[s][parameter]
            network.get_parameters()[parameter][network.hidden_neurons - network.n_non_learnable_neurons] +=\
                learning_rate * (accum_rewards[s] - baseline[network.hidden_neurons - network.n_non_learnable_neurons]) \
                * gradients[parameter][network.hidden_neurons - network.n_non_learnable_neurons]
            network.get_parameters()[parameter][network.output_neurons - network.n_non_learnable_neurons] +=\
                learning_rate * gradients[parameter][network.output_neurons - network.n_non_learnable_neurons]

    return baselines_num, baselines_den


def train_ml_batch(network, input_train, output_train, indices, learning_rate, beta, gamma, r):
    """"
    Train a network on the sequence passed as argument.
    """

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -network.n_output_neurons:, :, :]) == 0, \
        'There must be backward connections from output to hidden neurons.'

    network.set_mode('train_ml')

    training_sequence = torch.cat((input_train, output_train), dim=1)

    S_prime = input_train.shape[-1]
    potentials = []
    spiking_history = torch.zeros([network.n_neurons, network.alphabet_size, 1])
    rewards = []

    baselines_num = []
    baselines_den = []

    for j, sample in enumerate(indices):
        for s in range(S_prime):
            reward = feedforward_sampling_ml(network, training_sequence[sample], s, r, gamma)

            rewards.append(reward)
            potentials.append(network.potential)
            spiking_history = torch.cat((spiking_history, network.spiking_history[:, :, -1].unsqueeze(2)), dim=-1)
            baselines_num.append({parameter: network.gradients[parameter].pow(2) + 1e-12 for parameter in network.gradients})
            baselines_den.append({parameter: network.gradients[parameter].pow(2) + 1e-12 for parameter in network.gradients})

        # Do the batch update
        baselines_num, baselines_den = \
            batch_update_ml(network, j, rewards, baselines_num, baselines_den, spiking_history, potentials, learning_rate, S_prime, beta)

        # Reset learning variables
        rewards = []
        potentials = []
        spiking_history = torch.zeros([network.n_neurons, network.alphabet_size, 1])

        refractory_period(network)

        if j % int(len(indices) / 5) == 0:
            print('Sample %d out of %d' % (j, len(indices)))


def feedforward_sampling_rl(network, input_train, target, s, r, alpha, gamma):
    _ = network(input_train[:, :, s])

    spiking_indices_output = torch.max(torch.softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]),
                                                                network.potential[network.output_neurons - network.n_non_learnable_neurons, :]), dim=-1),
                                                     dim=-1), dim=-1)[1]

    probas = torch.softmax(torch.cat((torch.zeros([network.n_learnable_neurons, 1]), network.potential), dim=-1), dim=-1)

    reg = torch.sum(torch.sum(network.spiking_history[network.learnable_neurons, :, -1] * torch.log(1e-07 + probas[:, 1:] / ((1 - r) / network.alphabet_size)), dim=-1)
                    + (1 - torch.sum(network.spiking_history[network.learnable_neurons, :, -1], dim=-1)) * torch.log(1e-07 + probas[:, 0] / r))


    reward = custom_softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]), network.spiking_history[network.output_neurons, :, -1]),
                                      dim=-1)[[i for i in range(network.n_output_neurons)], spiking_indices_output], alpha)[target] \
             - torch.sum(custom_softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]), network.spiking_history[network.output_neurons, :, -1]),
                                                  dim=-1)[[i for i in range(network.n_output_neurons)],
                                                          spiking_indices_output], alpha)[[i for i in range(network.alphabet_size) if i not in target]])  \
             - gamma * reg

    return reward


def rl_update(network, eligibility_trace, reward, updates, baseline_num, baseline_den, baseline, learning_rate, kappa, beta):
    for parameter in updates:
        eligibility_trace[parameter] = time_average(eligibility_trace[parameter], network.gradients[parameter], kappa)

        baseline_num[parameter] = time_average(baseline_num[parameter], eligibility_trace[parameter].pow(2) * reward, beta)
        baseline_den[parameter] = time_average(baseline_den[parameter], eligibility_trace[parameter].pow(2), beta)
        baseline[parameter] = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        updates[parameter] = time_average(updates[parameter], (reward - baseline[parameter]) * eligibility_trace[parameter], kappa)

        network.get_parameters()[parameter] += updates[parameter] * learning_rate

    return baseline_num, baseline_den, baseline, updates, eligibility_trace



def init_policy_rl(network, input_train, target, learning_rate, alpha, beta, gamma, kappa, r):
    reward = feedforward_sampling_rl(network, input_train, target, 0, r, alpha, gamma)

    eligibility_trace = {parameter: network.gradients[parameter] for parameter in network.gradients}

    baseline_num = {parameter: eligibility_trace[parameter].pow(2) * reward for parameter in eligibility_trace}
    baseline_den = {parameter: eligibility_trace[parameter].pow(2) for parameter in eligibility_trace}

    baseline = {parameter: (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07) for parameter in network.gradients}

    updates = {parameter: (reward - baseline[parameter]) * eligibility_trace[parameter] for parameter in eligibility_trace}

    # Compute update
    for parameter in updates:
        network.get_parameters()[parameter] += updates[parameter] * learning_rate

    for s in range(1, input_train.shape[-1]):
        reward = feedforward_sampling_rl(network, input_train, target, s, r, alpha, gamma)
        baseline_num, baseline_den, baseline, updates, eligibility_trace = \
            rl_update(network, eligibility_trace, reward, updates, baseline_num, baseline_den, baseline, learning_rate, kappa, beta)

    return baseline_num, baseline_den, baseline, updates, eligibility_trace


def train_policy_based_rl_online(network, input_train, output_train, indices, learning_rate, kappa, alpha, beta, gamma, r):
    """"
    Train a network .
    """

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -network.n_output_neurons:, :, :]) > 0,\
        'There must be backward connections from output to hidden neurons.'
    network.set_mode('train_rl')

    S_prime = input_train.shape[-1]

    target = torch.max(torch.sum(output_train, dim=(-1, -2)), dim=-1).indices[indices[0]]
    baseline_num, baseline_den, baseline, updates, eligibility_trace = init_policy_rl(network, input_train[indices[0]], target, learning_rate, alpha, beta, gamma, kappa, r)

    for j, sample in enumerate(indices[1:]):
        # Reset network & training variables for each example
        refractory_period(network)

        target = torch.max(torch.sum(output_train, dim=(-1, -2)), dim=-1).indices[sample]
        for s in range(S_prime):
            reward = feedforward_sampling_rl(network, input_train[sample], target, s, r, alpha, gamma)
            baseline_num, baseline_den, baseline, updates, eligibility_trace = \
                rl_update(network, eligibility_trace, reward, updates, baseline_num, baseline_den, baseline, learning_rate, kappa, beta)


        if j % int(len(indices) / 5) == 0:
            # print(torch.sum(network.feedforward_weights, dim=(-1, -2, -3, -4)))
            print('Sample %d out of %d' % (j, len(indices)))


def batch_update_rl(network, sample, rewards, baselines_num, baselines_den, spiking_history, potentials, learning_rate, S_prime, beta):
    assert len(rewards) == S_prime
    assert len(potentials) == S_prime

    accum_rewards = [0] * S_prime
    accum_rewards[-1] = rewards[-1]

    for s in range(S_prime - 2, -1, -1):
        accum_rewards[s] = rewards[s] + accum_rewards[s + 1]

    # Compute the baseline
    for s in range(S_prime):
        gradients = network.compute_gradients(spiking_history[network.learnable_neurons, :, s + 1], potentials[s],
                                              network.compute_ff_trace(spiking_history[:, :, max(0, s + 1 - network.memory_length): s + 1]),
                                              network.compute_fb_trace(spiking_history[:, :, max(0, s + 1 - network.memory_length): s + 1]))

        for parameter in gradients:
            if sample == 0:
                baselines_num[s][parameter] *= accum_rewards[s]
            else:
                baselines_num[s][parameter] = beta * baselines_num[s][parameter] + (1 - beta) * (gradients[parameter].pow(2) + 1e-12) * accum_rewards[s]
                baselines_den[s][parameter] = beta * baselines_den[s][parameter] + (1 - beta) * (gradients[parameter].pow(2) + 1e-12)

            baseline = baselines_num[s][parameter] / baselines_den[s][parameter]
            network.get_parameters()[parameter] += learning_rate * (accum_rewards[s] - baseline) * gradients[parameter]

    return baselines_num, baselines_den


def train_rl_batch(network, input_train, output_train, learning_rate, alpha, beta, gamma, r):
    """"
    Train a network on the sequence passed as argument.
    """

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -2:, :, :]) > 0, 'There must be backward connections from output' \
                                                                                                                            'to hidden neurons.'

    network.set_mode('train_rl')

    epochs = input_train.shape[0]
    S_prime = input_train.shape[-1]
    S = epochs * S_prime
    potentials = []
    spiking_history = torch.zeros([network.n_neurons, network.alphabet_size, 1])
    rewards = []
    target = torch.max(torch.sum(output_train, dim=(-1, -2)), dim=-1).indices[0]

    baselines_num = []
    baselines_den = []

    for s in range(S_prime):
        _ = network(input_train[int(s / S_prime), :, :, s % S_prime])

        probas = torch.softmax(torch.cat((torch.zeros([network.n_learnable_neurons, 1]), network.potential), dim=-1), dim=-1)

        reg = torch.sum(torch.sum(network.spiking_history[network.learnable_neurons, :, -1] * torch.log(1e-07 + probas[:, 1:] / ((1 - r) / network.alphabet_size)), dim=-1)
                        + (1 - torch.sum(network.spiking_history[network.learnable_neurons, :, -1], dim=-1)) * torch.log(1e-07 + probas[:, 0] / r))

        spiking_indices_output = torch.max(torch.softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]),
                                                                    network.potential[network.output_neurons - network.n_non_learnable_neurons, :]), dim=-1),
                                                         dim=-1), dim=-1)[1]

        reward = custom_softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]), network.spiking_history[network.output_neurons, :, -1]),
                                          dim=-1)[[i for i in range(network.n_output_neurons)], spiking_indices_output], alpha)[target] \
                 - torch.sum(custom_softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]), network.spiking_history[network.output_neurons, :, -1]),
                                            dim=-1)[[i for i in range(network.n_output_neurons)],
                                                    spiking_indices_output], alpha)[[i for i in range(network.alphabet_size) if i not in target]])

        rewards.append(reward)
        potentials.append(network.potential)
        spiking_history = torch.cat((spiking_history, network.spiking_history[:, :, -1].unsqueeze(2)), dim=-1)
        baselines_num.append({parameter: network.gradients[parameter].pow(2) + 1e-12 for parameter in network.gradients})
        baselines_den.append({parameter: network.gradients[parameter].pow(2) + 1e-12 for parameter in network.gradients})


    for s in range(S_prime, S):
        # Reset network & training variables for each example
        if s % S_prime == 0:
            network.reset_internal_state()

            # Do the batch update
            baselines_num, baselines_den = \
                batch_update_rl(network, int(s / S_prime) - 1, rewards, baselines_num, baselines_den, spiking_history, potentials, learning_rate, S_prime, beta)

            # Set up new target
            target = torch.max(torch.sum(output_train, dim=(-1, -2)), dim=-1).indices[int(s/S_prime)]

            # Reset learning variables
            rewards = []
            potentials = []
            spiking_history = torch.zeros([network.n_neurons, network.alphabet_size, 1])

        _ = network(input_train[int(s / S_prime), :, :, s % S_prime])

        spiking_indices_output = torch.max(torch.softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]),
                                                                    network.potential[network.output_neurons - network.n_non_learnable_neurons, :]), dim=-1),
                                                         dim=-1), dim=-1)[1]

        probas = torch.softmax(torch.cat((torch.zeros([network.n_learnable_neurons, 1]), network.potential), dim=-1), dim=-1)

        reg = torch.sum(torch.sum(network.spiking_history[network.learnable_neurons, :, -1] * torch.log(1e-07 + probas[:, 1:] / ((1 - r) / network.alphabet_size)), dim=-1)
                        + (1 - torch.sum(network.spiking_history[network.learnable_neurons, :, -1], dim=-1)) * torch.log(1e-07 + probas[:, 0] / r))


        reward = custom_softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]), network.spiking_history[network.output_neurons, :, -1]),
                                          dim=-1)[[i for i in range(network.n_output_neurons)], spiking_indices_output], alpha)[target] \
                 - torch.sum(custom_softmax(torch.cat((torch.zeros([network.n_output_neurons, 1]), network.spiking_history[network.output_neurons, :, -1]),
                                            dim=-1)[[i for i in range(network.n_output_neurons)],
                                                    spiking_indices_output], alpha)[[i for i in range(network.alphabet_size) if i not in target]])

        rewards.append(reward)
        potentials.append(network.potential)
        spiking_history = torch.cat((spiking_history, network.spiking_history[:, :, -1].unsqueeze(2)), dim=-1)


        if s % (S / 5) == 0:
            print('Step %d out of %d' % (s, S))
