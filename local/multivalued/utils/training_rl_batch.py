import torch
from data_preprocessing.misc import custom_softmax


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


def train(network, input_train, output_train, learning_rate, alpha, beta, gamma, r):
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
