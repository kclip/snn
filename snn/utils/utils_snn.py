import torch
import numpy as np
import pickle

from snn.data_preprocessing.load_data import *


def refractory_period(network):
    """"
    Neural refractory period between two samples
    """
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.visible_neurons)], dtype=torch.float).to(network.device))


def get_acc_and_loss(network, hdf5_group, test_indices, T, n_classes, input_shape, dt, x_max, polarity):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    if len(test_indices) > len(hdf5_group.labels):
        test_indices = np.random.choice(np.arange(len(hdf5_group.labels)), [len(hdf5_group.labels)], replace=False)

    outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])
    loss = 0

    true_classes = torch.LongTensor(hdf5_group.labels[test_indices, 0])

    for j, idx in enumerate(test_indices):
        refractory_period(network)

        inputs, lbl = get_example(hdf5_group, idx, T, n_classes, input_shape, dt, x_max, polarity)
        inputs = inputs.to(network.device)

        for t in range(T):
            log_proba = network(inputs[:, t])
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[j, :, t] = network.spiking_history[network.output_neurons, -1].cpu()

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss


def get_acc_loss_and_spikes(network, hdf5_group, test_indices, T, n_classes, input_shape, dt, x_max, polarity):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    spikes = torch.zeros([len(test_indices), network.n_learnable_neurons, T])
    outputs = torch.zeros([len(test_indices), network.n_output_neurons, T])
    loss = 0

    true_classes = torch.LongTensor(hdf5_group.labels[test_indices, 0])

    for j, idx in enumerate(test_indices):
        refractory_period(network)

        inputs, lbl = get_example(hdf5_group, idx, T, n_classes, input_shape, dt, x_max, polarity)
        inputs = inputs.to(network.device)

        for t in range(T):
            log_proba = network(inputs[:, t])
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[j, :, t] = network.spiking_history[network.output_neurons, -1].cpu()
            spikes[j, :, t] = network.spiking_history[network.learnable_neurons, -1].cpu()

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss, spikes


def test(network, j, train_data, train_indices, test_data, test_indices, T, n_classes, input_shape,
         dt, x_max, polarity, test_period, train_accs, train_losses, test_accs, test_losses, save_path):
    if (j + 1) % test_period == 0:
        if (test_accs is not None) or (test_losses is not None):
            test_acc, test_loss = get_acc_and_loss(network, test_data, test_indices, T, n_classes, input_shape, dt, x_max, polarity)

            if test_accs is not None:
                test_accs[int(j + 1)].append(test_acc)
                print('test accuracy at ite %d: %f' % (int(j + 1), test_acc))
                with open(save_path + '/test_accs.pkl', 'wb') as f:
                    pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)

            if test_losses is not None:
                test_losses[int(j + 1)].append(test_loss)
                with open(save_path + '/test_losses.pkl', 'wb') as f:
                    pickle.dump(test_losses, f, pickle.HIGHEST_PROTOCOL)

        if (train_accs is not None) or (train_losses is not None):
            train_acc, train_loss = get_acc_and_loss(network, train_data, train_indices, T, n_classes, input_shape, dt, x_max, polarity)

            if train_accs is not None:
                train_accs[int(j + 1)].append(train_acc)
                with open(save_path + '/train_accs.pkl', 'wb') as f:
                    pickle.dump(train_accs, f, pickle.HIGHEST_PROTOCOL)

            if train_losses is not None:
                train_losses[int(j + 1)].append(train_loss)
                with open(save_path + '/train_losses.pkl', 'wb') as f:
                    pickle.dump(train_losses, f, pickle.HIGHEST_PROTOCOL)

        network.save(save_path + '/network_weights.hdf5')

        network.train()
        network.reset_internal_state()



