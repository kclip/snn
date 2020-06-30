import torch
import pickle
from binary_snn.utils_binary.misc import refractory_period, get_acc_and_loss, get_train_acc_and_loss


def feedforward_sampling(network, example, args):
    """"
    Feedforward sampling step:
    - passes information through the network
    - computes the learning signal
    """

    log_proba = network(example)

    # Accumulate learning signal
    proba_hidden = torch.sigmoid(network.potential[network.hidden_neurons - network.n_input_neurons])
    ls = torch.sum(log_proba[network.output_neurons - network.n_input_neurons]) \
          - args.alpha * torch.sum(network.spiking_history[network.hidden_neurons, -1]
          * torch.log(1e-12 + proba_hidden / args.r)
          + (1 - network.spiking_history[network.hidden_neurons, -1]) * torch.log(1e-12 + (1. - proba_hidden) / (1 - args.r)))

    return log_proba, ls


def local_feedback_and_update(network, ls_tmp, eligibility_trace_hidden, eligibility_trace_output,
                              learning_signal, baseline_num, baseline_den, args):
    """"
    Runs the local feedback and update steps:
    - computes the learning signal
    - updates the learning parameter
    """

    # local feedback
    if ls_tmp != 0:
        learning_signal = args.kappa * learning_signal + (1 - args.kappa) * ls_tmp

    # Update parameter
    for parameter in network.gradients:
        eligibility_trace_hidden[parameter].mul_(args.kappa).add_(1 - args.kappa, network.gradients[parameter][network.hidden_neurons - network.n_input_neurons])

        baseline_num[parameter].mul_(args.beta).add_(1 - args.beta, eligibility_trace_hidden[parameter].pow(2).mul_(learning_signal))
        baseline_den[parameter].mul_(args.beta).add_(1 - args.beta, eligibility_trace_hidden[parameter].pow(2))
        baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

        network.get_parameters()[parameter][network.hidden_neurons - network.n_input_neurons] \
            += args.lr * (learning_signal - baseline) * eligibility_trace_hidden[parameter]

        if eligibility_trace_output is not None:
            eligibility_trace_output[parameter].mul_(args.kappa).add_(1 - args.kappa, network.gradients[parameter][network.output_neurons - network.n_input_neurons])
            network.get_parameters()[parameter][network.output_neurons - network.n_input_neurons] += args.lr * eligibility_trace_output[parameter]

    return eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den


def init_training(network, args):
    network.set_mode('train')

    eligibility_trace_hidden = {parameter: network.gradients[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.gradients}
    eligibility_trace_output = {parameter: network.gradients[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.gradients}

    learning_signal = 0

    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

    S_prime = args.dataset.root.stats.train_label[:][-1]

    return eligibility_trace_output, eligibility_trace_hidden, learning_signal, baseline_num, baseline_den, S_prime


def train(network, indices, test_indices, args):
    """"
    Train a network.
    """

    eligibility_trace_output, eligibility_trace_hidden, learning_signal, baseline_num, baseline_den, S_prime = init_training(network, args)

    for j, idx in enumerate(indices[args.start_idx:]):
        j += args.start_idx
        if (j + 1) % 5 * (args.dataset.root.train.data[:].shape[0]) == 0:
            args.lr /= 2

        # Regularly test the accuracy
        if args.test_accs:
            if (j + 1) in args.test_accs:
                acc, loss = get_acc_and_loss(network, args.dataset, test_indices)
                args.test_accs[int(j + 1)].append(acc)
                print('test accuracy at ite %d: %f' % (int(j + 1), acc))

                # acc_train, _ = get_train_acc_and_loss(network, dataset, args.labels)
                # print('train accuracy at ite %d: %f' % (int(j + 1), acc_train))

                if args.save_path is not None:
                    with open(args.save_path + '/test_accs.pkl', 'wb') as f:
                        pickle.dump(args.test_accs, f, pickle.HIGHEST_PROTOCOL)

                    network.save(args.save_path + '/network_weights.hdf5')

                network.set_mode('train')

        refractory_period(network)
        sample = torch.cat((torch.FloatTensor(args.dataset.root.train.data[idx]),
                            torch.FloatTensor(args.dataset.root.train.label[idx])), dim=0).to(network.device)

        for s in range(S_prime):
            # Feedforward sampling
            log_proba, ls_tmp = feedforward_sampling(network, sample[:, s], args)
            # Local feedback and update
            eligibility_trace_hidden, eligibility_trace_output, learning_signal, baseline_num, baseline_den \
                = local_feedback_and_update(network, ls_tmp, eligibility_trace_hidden, eligibility_trace_output,
                                            learning_signal, baseline_num, baseline_den, args)

        if j % max(1, int(len(indices) / 5)) == 0:
            print('Step %d out of %d' % (j, len(indices)))

    return args.test_accs
