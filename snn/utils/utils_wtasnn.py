import torch
import pickle
import os


def refractory_period(network):
    length = network.memory_length + 1
    for s in range(length):
        network(torch.zeros([len(network.input_neurons), network.alphabet_size], dtype=torch.float).to(network.device),
                torch.zeros([len(network.output_neurons), network.alphabet_size], dtype=torch.float).to(network.device))


def get_acc_and_loss(network, dataloader, n_examples, T):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.eval()
    network.reset_internal_state()

    iterator = iter(dataloader)
    outputs = torch.zeros([n_examples, network.n_output_neurons, network.alphabet_size, T])

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
        inputs = inputs[0].transpose(1, 2).to(network.device)

        for t in range(T):
            log_proba = network(inputs[t].to(network.device))
            loss += torch.sum(log_proba).cpu().numpy()
            outputs[ite, :, :, t] = network.spiking_history[network.output_neurons, :, -1].cpu()

    predictions = torch.max(torch.sum(outputs, dim=(-1, -2)), dim=-1).indices
    true_classes = torch.max(torch.sum(true_classes, dim=(-1, -2)), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss


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
