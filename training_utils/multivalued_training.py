import torch
from multivalued_snn.utils_multivalued.misc import refractory_period, get_acc_and_loss
import pickle
from data_preprocessing.load_data import get_example
import os


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


def init_training(network):
    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_input_neurons, :, -network.n_output_neurons:]) == 0,\
        'There must be no backward connection from output to hidden neurons.'
    network.set_mode('train_ml')


    eligibility_trace_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.gradients}
    eligibility_trace_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.gradients}
    updates_hidden = {parameter: eligibility_trace_hidden[parameter] for parameter in network.gradients}

    reward = 0
    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2)*reward for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}


    return eligibility_trace_output, eligibility_trace_hidden, updates_hidden, baseline_num, baseline_den, reward


def train(network, dataset, sample_length, dt, input_shape, polarity, indices, test_indices, lr, n_classes, r, beta, gamma, kappa, start_idx, test_accs, save_path):

    eligibility_trace_output, eligibility_trace_hidden, updates_hidden, baseline_num, baseline_den, reward = init_training(network)

    train_data = dataset.root.train
    test_data = dataset.root.test
    T = int(sample_length * 1000 / dt)


    for j, idx in enumerate(indices[start_idx:]):
        j += start_idx
        if (j + 1) % (3 * (dataset.root.stats.train_data[0])) == 0:
            lr /= 2

        if test_accs:
            if (j + 1) in test_accs:
                acc, loss = get_acc_and_loss(network, test_data, test_indices, T, n_classes, input_shape, dt, dataset.root.stats.train_data[1], polarity)
                test_accs[int(j + 1)].append(acc)
                print('test accuracy at ite %d: %f' % (int(j + 1), acc))

                if save_path is not None:
                    with open(save_path + '/test_accs.pkl', 'wb') as f:
                        pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)
                    network.save(save_path + '/network_weights.hdf5')

                network.set_mode('train_ml')
                network.reset_internal_state()

        refractory_period(network)

        inputs, label = get_example(train_data, idx, T, n_classes, input_shape, dt, dataset.root.stats.train_data[1], polarity)
        sample = torch.cat((inputs, label), dim=0).to(network.device)

        for t in range(T):
            reward = feedforward_sampling_ml(network, sample[:, :, t].to(network.device), r, gamma)
            baseline_num, baseline_den, updates_hidden, eligibility_trace_hidden, eligibility_trace_output = \
                ml_update(network, eligibility_trace_hidden, eligibility_trace_output, reward, updates_hidden, baseline_num, baseline_den, lr, kappa, beta)

        if j % max(1, int(len(indices) / 5)) == 0:
            print('Sample %d out of %d' % (j + 1, len(indices)))

    # At the end of training, save final weights if none exist or if this ite was better than all the others
    if not os.path.exists(save_path + '/network_weights_final.hdf5'):
        network.save(save_path + '/network_weights_final.hdf5')
    else:
        if test_accs[list(test_accs.keys())[-1]][-1] >= max(test_accs[list(test_accs.keys())[-1]][:-1]):
            network.save(save_path + '/network_weights_final.hdf5')

    return test_accs
