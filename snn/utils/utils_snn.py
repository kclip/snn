import torch
import pickle
import os
import numpy as np
from snn.models.SNN import BinarySNN, LayeredSNN

def refractory_period(network):
    """"
    Neural refractory period between two samples
    """
    if isinstance(network, BinarySNN):
        length = network.memory_length + 1
        for t in range(length):
            network(torch.zeros([len(network.input_neurons)], dtype=torch.float).to(network.device),
                    torch.zeros([len(network.output_neurons)], dtype=torch.float).to(network.device))
    elif isinstance(network, LayeredSNN):
        length = np.max([l.memory_length for l in network.hidden_layers] + [network.out_layer.memory_length]) + 1
        refractory_sig = torch.zeros([network.batch_size, network.n_input_neurons, length]).to(network.device)
        for t in range(length):
            network(refractory_sig[:, :, :(t+1)])



def get_acc_and_loss(network, dataloader, n_examples, T):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    iterator = iter(dataloader)
    outputs = torch.zeros([n_examples, network.n_output_neurons, T])
    loss = 0

    hidden_hist = torch.zeros([n_examples, network.n_hidden_neurons, T])

    true_classes = torch.FloatTensor()

    for ite in range(n_examples):
        refractory_period(network)

        try:
            inputs, lbls = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            inputs, lbls = next(iterator)

        true_classes = torch.cat((true_classes, lbls), dim=0)
        inputs = inputs.to(network.device)

        for t in range(T):
            log_proba = network(inputs[:, t])
            loss += torch.sum(log_proba).cpu().numpy()

            outputs[ite, :, t] = network.spiking_history[network.output_neurons, -1].cpu()
            hidden_hist[ite, :, t] = network.spiking_history[network.hidden_neurons, -1].cpu()

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(true_classes, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float)) / len(predictions)

    print(torch.mean(hidden_hist))
    print(torch.mean(outputs))

    return acc, loss


def get_acc_loss_and_spikes(network, dataloader, n_examples, T):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    iterator = iter(dataloader)
    spikes = torch.zeros([n_examples, network.n_learnable_neurons, T])
    outputs = torch.zeros([n_examples, network.n_output_neurons, T])
    loss = 0

    true_classes = torch.FloatTensor()

    for ite in range(n_examples):
        refractory_period(network)

        try:
            inputs, lbls = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            inputs, lbls = next(iterator)
        true_classes = torch.cat((true_classes, lbls), dim=0)
        inputs = inputs.to(network.device)

        for t in range(T):
            log_proba = network(inputs[:, t])
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[ite, :, t] = network.spiking_history[network.output_neurons, -1].cpu()
            spikes[ite, :, t] = network.spiking_history[network.learnable_neurons, -1].cpu()

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(true_classes, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float)) / len(predictions)

    return acc, loss, spikes


def get_acc_layered(network, dataloader, T):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()

    iterator = iter(dataloader)
    outputs = torch.FloatTensor()

    true_classes = torch.FloatTensor()

    for inputs, lbls in iterator:
        if len(inputs) == network.batch_size: #todo fix non standard batch sizes
            refractory_period(network)

            true_classes = torch.cat((true_classes, lbls), dim=0)

            inputs = inputs.to(network.device).transpose(1, 2)
            outputs_batch = torch.zeros_like(lbls[:, :, 0])
            for t in range(T):
                network(inputs[:, :, :(t+1)])
                outputs_batch += network.out_layer.spiking_history[:, :, -1].cpu()

            outputs = torch.cat((outputs, outputs_batch))

    predictions = outputs.argmax(-1)
    true_classes = torch.max(torch.sum(true_classes, dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float)) / len(predictions)

    print(torch.mean(outputs))

    return acc


def test(network, params, ite, train_dl, T_train, test_dl, T_test, test_period, train_accs, train_losses, test_accs, test_losses, save_path):
    if (ite + 1) % test_period == 0:
        if (test_accs is not None) or (test_losses is not None):
            test_acc, test_loss = get_acc_and_loss(network, test_dl, params['n_examples_test'], T_test)

            if test_accs is not None:
                test_accs[int(ite + 1)].append(test_acc)
                print('test accuracy at ite %d: %f' % (int(ite + 1), test_acc))
                with open(save_path + '/test_accs.pkl', 'wb') as f:
                    pickle.dump(test_accs, f, pickle.HIGHEST_PROTOCOL)

                if not os.path.exists(save_path + '/network_weights_best.hdf5'):
                    network.save(save_path + '/network_weights_best.hdf5')
                else:
                    if test_acc >= max([max(j) for j in test_accs.values() if len(j)>0]):
                        network.save(save_path + '/network_weights_best.hdf5')

            if test_losses is not None:
                test_losses[int(ite + 1)].append(test_loss)
                with open(save_path + '/test_losses.pkl', 'wb') as f:
                    pickle.dump(test_losses, f, pickle.HIGHEST_PROTOCOL)

        if (train_accs is not None) or (train_losses is not None):
            train_acc, train_loss = get_acc_and_loss(network, train_dl, params['n_examples_train'], T_train)

            if train_accs is not None:
                train_accs[int(ite + 1)].append(train_acc)
                with open(save_path + '/train_accs.pkl', 'wb') as f:
                    pickle.dump(train_accs, f, pickle.HIGHEST_PROTOCOL)

            if train_losses is not None:
                train_losses[int(ite + 1)].append(train_loss)
                with open(save_path + '/train_losses.pkl', 'wb') as f:
                    pickle.dump(train_losses, f, pickle.HIGHEST_PROTOCOL)

        if not os.path.exists(save_path + '/network_weights_best.hdf5'):
            network.save(save_path + '/network_weights.hdf5')

        network.train()
        network.reset_internal_state()

    return train_accs, train_losses, test_accs, test_losses



