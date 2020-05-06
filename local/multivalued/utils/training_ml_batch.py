import torch
from data_preprocessing.misc import refractory_period


def feedforward_sampling_ml(network, training_sequence, r, gamma):
    log_proba = network(training_sequence)

    probas = torch.softmax(torch.cat((torch.zeros([network.n_hidden_neurons, 1]).to(network.device), network.potential[network.hidden_neurons - network.n_non_learnable_neurons]), dim=-1), dim=-1)

    reg = torch.sum(torch.sum(network.spiking_history[network.hidden_neurons, :, -1] * torch.log(1e-07 + probas[:, 1:] / ((1 - r) / network.alphabet_size)), dim=-1)
                    + (1 - torch.sum(network.spiking_history[network.hidden_neurons, :, -1], dim=-1)) * torch.log(1e-07 + probas[:, 0] / r))

    reward = torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons]) - gamma * reg

    return reward


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


def train(network, input_train, output_train, indices, learning_rate, beta, gamma, r):
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
