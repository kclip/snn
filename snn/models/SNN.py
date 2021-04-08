from __future__ import print_function
import torch
import numpy as np

from snn.utils import filters
from snn.models.base import SNNetwork, SNNLayer


class BinarySNN(SNNetwork):
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, topology, synaptic_filter=filters.base_filter,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=1, initialization='uniform', weights_magnitude=0.01, device='cpu', save_path=None):

        super(BinarySNN, self).__init__(n_input_neurons=n_input_neurons, n_hidden_neurons=n_hidden_neurons, n_output_neurons=n_output_neurons, topology=topology,
                                        synaptic_filter=synaptic_filter, n_basis_feedforward=n_basis_feedforward, n_basis_feedback=n_basis_feedback, tau_ff=tau_ff,
                                        tau_fb=tau_fb, mu=mu, initialization=initialization, weights_magnitude=weights_magnitude, device=device, save_path=save_path)


        # Feedforward weights are a tensor of size [n_learnable_neurons, n_neurons, n_basis_feedforward] for which the block-diagonal elements are 0,
        # and otherwise 1s in the topology are distributed according to initialization
        self.ff_weights_shape = torch.Size([self.n_learnable_neurons, self.n_neurons, self.n_basis_feedforward])
        self.feedforward_mask = torch.tensor(np.kron(topology, np.ones([self.n_basis_feedforward])).reshape(self.ff_weights_shape), dtype=torch.float).to(self.device)
        assert self.feedforward_mask.shape == self.ff_weights_shape

        self.initialize_ff_weights(topology, howto=initialization, gain=weights_magnitude)


        # Feedback weights are a tensor of size [n_neurons, n_basis_feedback], for which learnable elements are distributed according to initialization
        self.fb_weights_shape = torch.Size([self.n_learnable_neurons, self.n_basis_feedback])
        self.initialize_fb_weights(topology, howto=initialization, gain=weights_magnitude)


        # Bias weights are a tensor of size [n_learnable_neurons], for which learnable elements are distributed according to initialization
        self.bias_shape = torch.Size([self.n_learnable_neurons])
        self.initialize_bias_weights(topology, howto=initialization, gain=weights_magnitude)

        # Number of timesteps to keep in memory
        self.memory_length = max(self.tau_ff, self.tau_fb)

        ### State of the network
        self.spiking_history = torch.zeros([self.n_neurons, 2]).to(self.device)



    def forward(self, input_signal, output_signal=None):
        assert self.n_neurons == (len(self.input_neurons) + len(self.hidden_neurons) + len(self.output_neurons)), "The numbers of neurons don't match"
        assert self.n_neurons == (len(self.learnable_neurons) + len(self.input_neurons)), "The numbers of neurons don't match"


        ### Compute potential
        ff_trace = self.compute_ff_trace(self.spiking_history[:, 1:])
        fb_trace = self.compute_fb_trace(self.spiking_history[:, 1:])[self.learnable_neurons, :]

        self.potential = self.compute_ff_potential(ff_trace) + self.compute_fb_potential(fb_trace) + self.bias

        ### Update spiking history
        self.spiking_history = self.update_spiking_history(input_signal, output_signal)

        ### Compute log-probabilities
        # noinspection PyTypeChecker
        log_proba = self.spiking_history[self.learnable_neurons, -1] * torch.log(1e-07 + torch.sigmoid(self.potential)) \
                    + (1 - self.spiking_history[self.learnable_neurons, -1]) * torch.log(1. + 1e-07 - torch.sigmoid(self.potential))  # We add 1e-07 for numerical stability of the log

        assert log_proba.shape == torch.Size([self.n_learnable_neurons]), \
            'Wrong log_probability shape, got: ' + str(log_proba.shape) + ', expected: ' + str(torch.Size([self.n_learnable_neurons]))


        ### Compute gradients
        if self.training:
            self.compute_gradients(self.spiking_history[self.learnable_neurons, -1], self.potential, ff_trace, fb_trace)

        return log_proba


    ### Weights initialization
    def initialize_ff_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / ((torch.sum(topology[:, i]) + torch.sum(topology[i, :])) * self.n_basis_feedforward) for i in range(self.n_learnable_neurons)]).flatten()
            std = std.unsqueeze(1).unsqueeze(2).repeat(1, self.n_neurons, self.n_basis_feedforward)

            assert std.shape == self.ff_weights_shape
            self.feedforward_weights = (torch.normal(gain * std, std).to(self.device) * self.feedforward_mask)
        elif howto == 'uniform':
            self.feedforward_weights = (gain * (torch.rand(self.ff_weights_shape) * 2 - 1).to(self.device) * self.feedforward_mask)

        self.ff_grad = torch.zeros(self.ff_weights_shape).to(self.device)

    def initialize_fb_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / ((torch.sum(topology[:, i]) + torch.sum(topology[i, :])) * self.n_basis_feedback) for i in range(self.n_learnable_neurons)]).flatten()
            std = std.unsqueeze(1).repeat(1, self.n_basis_feedback)

            assert std.shape == self.fb_weights_shape
            self.feedback_weights = (torch.normal(gain * std, std)).to(self.device)
        elif howto == 'uniform':
            self.feedback_weights = (gain * (torch.rand(self.fb_weights_shape) * 2 - 1)).to(self.device)

        self.fb_grad = torch.zeros(self.fb_weights_shape).to(self.device)

    def initialize_bias_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / (torch.sum(topology[:, i]) + torch.sum(topology[i, :])) for i in range(self.n_learnable_neurons)]).flatten()
            assert std.shape == self.bias_shape
            self.bias = (torch.normal(gain * std, std)).to(self.device)
        elif howto == 'uniform':
            self.bias = (gain * (torch.rand(self.bias_shape) * 2 - 1)).to(self.device)

        self.bias_grad = torch.zeros(self.bias_shape).to(self.device)



    ### Computations
    def compute_ff_trace(self, spikes):
        return torch.matmul(spikes.flip(-1), self.feedforward_filter[:spikes.shape[-1]])


    def compute_fb_trace(self, spikes):
        return torch.matmul(spikes.flip(-1), self.feedback_filter[:spikes.shape[-1]])


    def compute_ff_potential(self, ff_trace):
        return torch.sum(self.feedforward_weights * ff_trace * self.feedforward_mask, dim=(-1, -2))


    def compute_fb_potential(self, fb_trace):
        return torch.sum(self.feedback_weights * fb_trace, dim=(-1))


    def generate_spikes(self, spiking_history, neurons_group):
        spiking_history[neurons_group, -1] = torch.bernoulli(torch.sigmoid(self.potential[neurons_group - self.n_input_neurons])).to(self.device)

        if torch.isnan(spiking_history).any():
            print('Spiking history')
            print(self.spiking_history[neurons_group, -1])
            print('Inputs')
            print(self.spiking_history[self.input_neurons, -5:])
            print('Potential')
            print(self.potential[neurons_group - self.n_input_neurons])

            raise RuntimeError

        return spiking_history

    def update_spiking_history(self, input_signal, output_signal=None):
        spiking_history = torch.cat((self.spiking_history[:, - self.memory_length:], torch.zeros([self.n_neurons, 1]).to(self.device)), dim=-1)
        spiking_history[self.input_neurons, -1] = input_signal

        if self.n_hidden_neurons > 0:
            spiking_history = self.generate_spikes(spiking_history, self.hidden_neurons)
        if output_signal is not None:
            spiking_history[self.output_neurons, -1] = output_signal
        else:
            spiking_history = self.generate_spikes(spiking_history, self.output_neurons)

        return spiking_history


    def compute_gradients(self, spikes, potential, feedforward_trace, feedback_trace):
        self.bias_grad = spikes - torch.sigmoid(potential)
        assert self.bias_grad.shape == self.bias.shape, "Wrong bias gradient shape"

        self.ff_grad = feedforward_trace.unsqueeze(0).repeat(self.n_learnable_neurons, 1, 1) \
                      * self.bias_grad.unsqueeze(1).repeat(1, self.n_neurons).unsqueeze(2).repeat(1, 1, self.n_basis_feedforward) \
                      * self.feedforward_mask
        assert self.ff_grad.shape == self.ff_weights_shape, "Wrong feedforward weights gradient shape"

        self.fb_grad = feedback_trace * self.bias_grad.unsqueeze(1).repeat(1, self.n_basis_feedback)
        assert self.fb_grad.shape == self.fb_weights_shape, "Wrong feedback weights gradient shape"





class LayeredSNN(torch.nn.Module):
    def __init__(self, n_input_neurons, n_neurons_per_layer, n_output_neurons, synaptic_filter=filters.base_filter, n_basis_feedforward=[8],
                 n_basis_feedback=[1], tau_ff=[10], tau_fb=[10], mu=[0.5], device='cpu'):

        super(LayeredSNN, self).__init__()
        '''
        '''

        self.device = device

        ### Network parameters
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = np.sum(n_neurons_per_layer)
        self.n_hidden_layers = len(n_neurons_per_layer)
        self.n_output_neurons = n_output_neurons
        self.n_neurons = n_input_neurons + self.n_hidden_neurons + self.n_output_neurons

        if len(n_basis_feedforward) == 1:
            n_basis_feedforward = n_basis_feedforward * (1 + self.n_hidden_layers)
        if len(n_basis_feedback) == 1:
            n_basis_feedback = n_basis_feedback * (1 + self.n_hidden_layers)
        if len(tau_ff) == 1:
            tau_ff = tau_ff * (1 + self.n_hidden_layers)
        if len(tau_fb) == 1:
            tau_fb = tau_fb * (1 + self.n_hidden_layers)
        if len(mu) == 1:
            mu = mu * (1 + self.n_hidden_layers)

        self.hidden_layers = torch.nn.ModuleList()
        Nhid = [n_input_neurons] + n_neurons_per_layer

        for i in range(self.n_hidden_layers):
            self.hidden_layers.append(SNNLayer(Nhid[i], Nhid[i + 1], synaptic_filter=synaptic_filter, n_basis_feedforward=n_basis_feedforward[i],
                                               n_basis_feedback=n_basis_feedback[i], tau_ff=tau_ff[i], tau_fb=tau_fb[i], mu=mu[i], device=device))

        self.out_layer = SNNLayer(Nhid[-1], n_output_neurons, synaptic_filter=synaptic_filter, n_basis_feedforward=n_basis_feedforward[-1],
                                  n_basis_feedback=n_basis_feedback[-1], tau_ff=tau_ff[-1], tau_fb=tau_fb[-1], mu=mu[-1], device=device)

        self.training = None

    def forward(self, inputs_history, target=None, n_samples=1):
        probas_hidden = torch.Tensor().to(self.device)
        outputs_hidden = torch.Tensor().to(self.device)

        net_probas = torch.Tensor().to(self.device)
        net_outputs = torch.Tensor().to(self.device)

        for i in range(n_samples):
            if self.n_hidden_layers > 0:
                probas_hidden_tmp = torch.Tensor()
                outputs_hidden_tmp = torch.Tensor()

                proba_layer, layer_outputs = self.hidden_layers[0](inputs_history, target=None, no_update=n_samples - 1 - i)
                probas_hidden_tmp = torch.cat((probas_hidden_tmp, proba_layer.unsqueeze(0)))
                outputs_hidden_tmp = torch.cat((outputs_hidden_tmp, layer_outputs.unsqueeze(0)))

                for j in range(1, self.n_hidden_layers):
                    proba_layer, layer_outputs = self.hidden_layers[j](self.hidden_layers[j - 1].spiking_history, target=None, no_update=n_samples - 1 - i)
                    probas_hidden_tmp = torch.cat((probas_hidden_tmp, proba_layer.unsqueeze(0)))
                    outputs_hidden_tmp = torch.cat((outputs_hidden_tmp, layer_outputs.unsqueeze(0)))

                probas_hidden = torch.cat((probas_hidden, probas_hidden_tmp.unsqueeze(0)))
                outputs_hidden = torch.cat((outputs_hidden, outputs_hidden_tmp.unsqueeze(0)))

                probas_output_tmp, net_output_tmp = self.out_layer(self.hidden_layers[-1].spiking_history, target, no_update=n_samples - 1 - i)

            else:
                probas_output_tmp, net_output_tmp = self.out_layer(inputs_history, target, no_update=n_samples - 1 - i)
                probas_hidden = None
                outputs_hidden = None
            
            net_probas = torch.cat((net_probas, probas_output_tmp.unsqueeze(0)))
            net_outputs = torch.cat((net_outputs, net_output_tmp.unsqueeze(0)))


        return net_probas, net_outputs, probas_hidden, outputs_hidden


    ### Setters
    def reset_weights(self):
        for l in self.hidden_layers:
            l.reset_weights()

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def eval(self):
        self.training = False

    ### Misc
    def save(self, path=None):
        # todo
        return

    def import_weights(self, path):
        # todo
        return

