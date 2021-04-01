import torch
from snn.models.SNN import LayeredSNN
import snn.utils.filters as filters
import numpy as np
from snn.models.SNN import BinarySNN
from snn.utils.misc import make_network_parameters, str2bool
from snn.training_utils.snn_training import feedforward_sampling, local_feedback_and_update
from snn.utils.utils_snn import refractory_period
from snn.optimizer.snnsgd import SNNSGD
import tables
from neurodata.load_data import create_dataloader
from snn.utils.utils_snn import get_acc_and_loss
from snn.utils.utils_snn import get_acc_layered

sample_length = 2000000  # length of samples during training in ms
dt = 25000  # us
polarity = False
T = int(sample_length / dt)  # number of timesteps in a sample
input_size = (1 + polarity) * 26 * 26

dataset_path = r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects\datasets\mnist-dvs\mnist_dvs_events_new.hdf5"
ds = 1

dataset = tables.open_file(dataset_path)
x_max = dataset.root.stats.train_data[1] // ds
dataset.close()

n_outputs = 2
n_hidden = 16
n_neurons_per_layer = [n_hidden]

network = LayeredSNN(input_size, n_neurons_per_layer, n_outputs,  synaptic_filter=filters.raised_cosine_pillow_08, n_basis_feedforward=[8],
                     n_basis_feedback=[1], tau_ff=[10], tau_fb=[10], mu=[0.5], device='cpu')

topology = torch.zeros([n_hidden + n_outputs, n_hidden + input_size + n_outputs])
topology[:n_hidden, :input_size] = 1
topology[n_hidden:, input_size:-n_outputs] = 1

network2 = BinarySNN(**make_network_parameters(network_type='snn',
                                               n_input_neurons=input_size,
                                               n_output_neurons=n_outputs,
                                               n_hidden_neurons=n_hidden,
                                               topology_type='custom',
                                               topology=topology
                                               ))

lr = 0.01
n_samples = 3

optimizer = SNNSGD([{'params': network.out_layer.parameters(), 'ls': False, 'baseline': False},
                    {'params': network.hidden_layers.parameters(), 'ls': True, 'baseline': True}
                    ], lr=lr)

loss_fn = torch.nn.BCELoss(reduction='mean')

network.train()
network2.train()

eligibility_trace_hidden = {parameter: network2.get_gradients()[parameter][network2.hidden_neurons - network2.n_input_neurons] for parameter in network2.get_gradients()}
eligibility_trace_output = {parameter: network2.get_gradients()[parameter][network2.output_neurons - network2.n_input_neurons] for parameter in network2.get_gradients()}

learning_signal = 0

baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden}
baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

classes = [1, 7]
train_dl, test_dl = create_dataloader(dataset_path, batch_size=1, size=[input_size], classes=classes,
                                      sample_length_train=sample_length, sample_length_test=sample_length, dt=dt,
                                      polarity=polarity, ds=ds, shuffle_test=True, num_workers=0)

r = 0.1
gamma = 1

train_iterator = iter(train_dl)

for ite in range(500):
    if (ite+1) % 50 == 0:
        print('Ite %d: ' % (ite+1))
        acc_layered = get_acc_layered(network, test_dl, 100, T)
        print('Acc with LayeredSNN', acc_layered)

        acc_snn, _ = get_acc_and_loss(network2, test_dl, 100, T)
        print('Acc with BinarySNN', acc_snn)

    network.train()
    network2.train()

    refractory_period(network)
    refractory_period(network2)

    try:
        inputs, targets = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_dl)
        inputs, targets = next(train_iterator)

    inputs = inputs[0].to(network.device)
    targets = targets[0].to(network.device)

    for t in range(T):
        ### LayeredSNN
        net_probas, net_outputs, probas_hidden, outputs_hidden = network(inputs[:t].T, targets[:, t], n_samples=n_samples)

        # Generate gradients and KL regularization for hidden neurons
        out_loss = loss_fn(net_probas, net_outputs)
        if probas_hidden is not None:
            hidden_loss = loss_fn(probas_hidden, outputs_hidden.detach())
            with torch.no_grad():
                kl_reg = - gamma * torch.mean(outputs_hidden * torch.log(1e-7 + probas_hidden / r)
                                              + (1 - outputs_hidden) * torch.log(1e-7 + (1 - probas_hidden) / (1 - r)))
        else:
            hidden_loss = 0
            kl_reg = 0

        loss = out_loss + hidden_loss
        loss.backward()

        optimizer.step(out_loss.detach() + kl_reg)
        optimizer.zero_grad()

        ### Binary SNN
        log_proba = network2(inputs[t],  targets[:, t])
        # Accumulate learning signal
        proba_hidden = torch.sigmoid(network2.potential[network2.hidden_neurons - network2.n_input_neurons])
        ls_tmp = torch.sum(log_proba[network2.output_neurons - network2.n_input_neurons]) \
                 - gamma * torch.sum(network2.spiking_history[network2.hidden_neurons, -1] * torch.log(1e-07 + proba_hidden / r)
                                     + (1 - network2.spiking_history[network2.hidden_neurons, -1]) * torch.log(1e-07 + (1. - proba_hidden) / (1 - r)))

        if ls_tmp != 0:
            learning_signal = 0.01 * learning_signal + (1 - 0.01) * ls_tmp

        # Update parameter
        for parameter in network2.get_gradients():
            eligibility_trace_hidden[parameter].mul_(0.01).add_(network2.get_gradients()[parameter][network2.hidden_neurons - network2.n_input_neurons], alpha=1 - 0.01)

            baseline_num[parameter].mul_(0.01).add_(eligibility_trace_hidden[parameter].pow(2).mul(learning_signal), alpha=1 - 0.01)
            baseline_den[parameter].mul_(0.01).add_(eligibility_trace_hidden[parameter].pow(2), alpha=1 - 0.01)
            baseline = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

            network2.get_parameters()[parameter][network2.hidden_neurons - network2.n_input_neurons] \
                += lr * learning_signal * eligibility_trace_hidden[parameter]

            if eligibility_trace_output is not None:
                eligibility_trace_output[parameter].mul_(0.01).add_(network2.get_gradients()[parameter][network2.output_neurons - network2.n_input_neurons], alpha=1 - 0.01)
                network2.get_parameters()[parameter][network2.output_neurons - network2.n_input_neurons] += lr * eligibility_trace_output[parameter]


