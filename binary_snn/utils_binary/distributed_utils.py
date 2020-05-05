import numpy as np
import torch
import torch.distributed as dist
import math
from binary_snn.models.SNN import SNNetwork
import datetime
import data_preprocessing.misc


def init_processes(rank, world_size, backend, url, net_params, train_params, train_func):
    """"
    Initialize process group and launches training on the nodes
    """
    dist.init_process_group(backend=backend, init_method=url, timeout=datetime.timedelta(0, 360000), world_size=world_size, rank=rank)
    print('Process %d started' % rank)
    train_func(rank, world_size, net_params, train_params)
    return


def init_training(rank, num_nodes, nodes_group, dataset, labels, net_parameters):
    """"
    Initializes the different parameters for distributed training
    """
    # Initialize an SNN
    network = SNNetwork(**data_preprocessing.misc.make_network_parameters(**net_parameters))

    # At the beginning, the master node:
    # - transmits its weights to the workers
    # - distributes the samples among workers
    if rank == 0:
        # Initializing an aggregation list for future weights collection
        weights_list = [[torch.zeros(network.feedforward_weights.shape, dtype=torch.float) for _ in range(num_nodes)],
                        [torch.zeros(network.feedback_weights.shape, dtype=torch.float) for _ in range(num_nodes)],
                        [torch.zeros(network.bias.shape, dtype=torch.float) for _ in range(num_nodes)],
                        [torch.zeros(1, dtype=torch.float) for _ in range(num_nodes)]]
        indices_local = [i for i in range(dataset.root.train.label[:].shape[0])]
    else:
        weights_list = []

        # distribute training samples among nodes
        n_labels_per_node = int(len(labels) / (num_nodes - 1))
        local_labels = labels[(rank - 1) * n_labels_per_node: rank * n_labels_per_node]

        indices_local = data_preprocessing.misc.find_train_indices_for_labels(dataset, local_labels)
        print(rank, indices_local)

    dist.barrier(nodes_group)

    # Master node sends its weights
    for parameter in network.get_parameters():
        dist.broadcast(network.get_parameters()[parameter], 0, group=nodes_group)
    if rank == 0:
        print('Node 0 has shared its model and training data is partitioned among workers')

    # The nodes initialize their eligibility trace and learning signal
    learning_signal = 0
    ls_temp = 0
    eligibility_trace = {parameter: network.gradients[parameter] for parameter in network.gradients}
    et_temp = {parameter: network.gradients[parameter] for parameter in network.gradients}


    return network, indices_local, weights_list, eligibility_trace, et_temp, learning_signal, ls_temp


def distribute_samples(nodes, rank, dataset, eta, num_samples):
    """
    The master node (rank 0) randomly chooses and transmits samples indices to each device for training.
    Upon reception of their assigned samples, the nodes create their training dataset
    """

    if rank == 0:
        # Indices corresponding to each class
        indices_0 = np.asarray(torch.max(torch.sum(torch.FloatTensor(dataset.root.label[:]), dim=-1), dim=-1).indices == 0).nonzero()[0]
        indices_1 = np.asarray(torch.max(torch.sum(torch.FloatTensor(dataset.root.label[:]), dim=-1), dim=-1).indices == 1).nonzero()[0]

        assert len(indices_0) == len(indices_1)
        n_main_class = math.floor(num_samples * eta)
        n_secondary_class = num_samples - n_main_class
        assert (n_main_class + n_secondary_class) == num_samples

        # Randomly select samples for each worker
        indices_worker_0 = np.hstack((np.random.choice(indices_0, [n_main_class], replace=False), np.random.choice(indices_1, [n_secondary_class], replace=False)))
        np.random.shuffle(indices_worker_0)
        remaining_indices_0 = [i for i in indices_0 if i not in indices_worker_0]
        remaining_indices_1 = [i for i in indices_1 if i not in indices_worker_0]
        indices_worker_1 = np.hstack((np.random.choice(remaining_indices_0, [n_secondary_class], replace=False), np.random.choice(remaining_indices_1, [n_main_class], replace=False)))
        np.random.shuffle(indices_worker_1)

        assert len(indices_worker_0) == len(indices_worker_1)

        # Send samples to the workers
        indices = [torch.zeros([num_samples], dtype=torch.int), torch.IntTensor(indices_worker_0), torch.IntTensor(indices_worker_1)]
        indices_local = torch.zeros([num_samples], dtype=torch.int)
        dist.scatter(tensor=indices_local, src=0, scatter_list=indices, group=nodes)

        # Save samples sent to the workers at master to evaluate train loss and accuracy later
        indices_local = torch.IntTensor(np.hstack((indices_worker_0, indices_worker_1)))

    else:
        indices_local = torch.zeros([num_samples], dtype=torch.int)
        dist.scatter(tensor=indices_local, src=0, scatter_list=[], group=nodes)
        assert torch.sum(indices_local) != 0

    return indices_local


def global_update(nodes, rank, network, weights_list):
    """"
    Global update step for distributed learning.
    """

    for j, parameter in enumerate(network.get_parameters()):
        if rank != 0:
            dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=[], dst=0, group=nodes)
        else:
            dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=weights_list[j], dst=0, group=nodes)
            network.get_parameters()[parameter].data = torch.mean(torch.stack(weights_list[j][1:]), dim=0)
        dist.broadcast(network.get_parameters()[parameter], 0, group=nodes)


def global_update_subset(nodes, rank, network, weights_list, gradients_accum, n_weights_to_send):
    """"
    Global update step for distributed learning when transmitting only a subset of the weights.
    Each worker node transmits a tensor in which only the indices corresponding to the synapses with the largest n_weights_to_send accumulated gradients are kept nonzero
    """

    for j, parameter in enumerate(network.get_parameters()):
        if j == 0:
            if rank != 0:
                to_send = network.get_parameters()[parameter].data  # each worker node copies its weights in a new vector
                # Selection of the indices to set to zero before transmission
                indices_not_to_send = [i for i in range(network.n_basis_feedforward) if i not in torch.topk(torch.sum(gradients_accum, dim=(0, 1)), n_weights_to_send)[1]]
                to_send[:, :, indices_not_to_send] = 0

                # Transmission of the quantized weights
                dist.gather(tensor=to_send, gather_list=[], dst=0, group=nodes)
            else:
                dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=weights_list[j], dst=0, group=nodes)

                indices_received = torch.bincount(torch.nonzero(torch.sum(torch.stack(weights_list[j][1:]), dim=(1, 2)))[:, 1])
                multiples = torch.zeros(network.n_basis_feedforward)  # indices of weights transmitted by two devices at once: those will be averaged
                multiples[:len(indices_received)] = indices_received
                multiples[multiples == 0] = 1

                # Averaging step
                network.get_parameters()[parameter].data = torch.sum(torch.stack(weights_list[j][1:]), dim=0) / multiples.type(torch.float)

        else:
            if rank != 0:
                dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=[], dst=0, group=nodes)
            else:
                dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=weights_list[j], dst=0, group=nodes)
                network.get_parameters()[parameter].data = torch.mean(torch.stack(weights_list[j][1:]), dim=0)
        dist.broadcast(network.get_parameters()[parameter], 0, group=nodes)
