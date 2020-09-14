from __future__ import print_function
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

from snn.utils.misc import custom_softmax
from snn.utils import filters
from snn.models.base import SNNetwork


class WTASNN(SNNetwork):
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, topology,
                 synaptic_filter=filters.base_filter, n_basis_feedforward=1, n_basis_feedback=1,
                 tau_ff=1, tau_fb=1, mu=1, weights_magnitude=0., initialization='glorot', device='cpu', save_path=None):

        super(WTASNN, self).__init__(n_input_neurons=n_input_neurons, n_hidden_neurons=n_hidden_neurons, n_output_neurons=n_output_neurons, topology=topology,
                                     synaptic_filter=synaptic_filter, n_basis_feedforward=n_basis_feedforward, n_basis_feedback=n_basis_feedback, tau_ff=tau_ff,
                                     tau_fb=tau_fb, mu=mu, initialization=initialization, weights_magnitude=weights_magnitude, device=device, save_path=save_path)


        ### Alphabet
        self.alphabet_size = 2
        self.alphabet = [i for i in range(1, self.alphabet_size + 1)]


        # Feedforward weights are a tensor of size [n_learnable_neurons, alphabet_size, n_neurons, alphabet_size, n_basis_feedforward] for which the block-diagonal elements are 0,
        # and otherwise 1s in the topology are distributed according to initialization
        self.ff_weights_shape = torch.Size([self.n_learnable_neurons,  self.alphabet_size, self.n_neurons, self.alphabet_size, self.n_basis_feedforward])
        self.feedforward_mask = topology.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(1, self.alphabet_size, 1, self.alphabet_size, self.n_basis_feedforward).to(self.device)

        assert self.feedforward_mask.shape == self.ff_weights_shape
        assert torch.sum(self.feedforward_mask[[i for i in range(self.n_learnable_neurons)], :, [i + self.n_input_neurons for i in range(self.n_learnable_neurons)], :]) == 0

        self.initialize_ff_weights(topology, howto=initialization, gain=weights_magnitude)


        # Feedback weights are a tensor of size [n_neurons, alphabet_size, n_basis_feedback], for which learnable elements are distributed according to initialization
        self.fb_weights_shape = torch.Size([self.n_learnable_neurons, self.alphabet_size, self.n_basis_feedback])
        self.initialize_fb_weights(topology, howto=initialization, gain=weights_magnitude)


        ### Bias
        self.bias_shape = torch.Size([self.n_learnable_neurons, self.alphabet_size])
        self.initialize_bias_weights(topology, howto=initialization, gain=weights_magnitude)


        ### State of the network
        self.spiking_history = torch.zeros([self.n_neurons, self.alphabet_size, 2]).to(self.device)



    def forward(self, input_signal):
        assert self.n_neurons == (len(self.input_neurons) + len(self.hidden_neurons) + len(self.output_neurons)), "The numbers of neurons don't match"
        assert self.n_neurons == (len(self.learnable_neurons) + len(self.input_neurons)), "The numbers of neurons don't match"

        ### Compute potential
        ff_trace = self.compute_ff_trace(self.spiking_history[:, :, 1:])
        fb_trace = self.compute_fb_trace(self.spiking_history[:, :, 1:])[self.learnable_neurons, :]

        self.potential = self.compute_ff_potential(ff_trace) + self.compute_fb_potential(fb_trace) + self.bias

        ### Update spiking history
        self.update_spiking_history(input_signal)

        ### Compute log-probabilities
        # noinspection PyTypeChecker
        log_proba = torch.sum(torch.cat((1 - torch.sum(self.spiking_history[self.learnable_neurons, :, -1], dim=-1).unsqueeze(1),
                                         self.spiking_history[self.learnable_neurons, :, -1]), dim=-1) \
                              * torch.log_softmax(torch.cat((torch.zeros([self.n_learnable_neurons, 1]).to(self.device), self.potential), dim=-1), dim=-1),
                              dim=-1)

        assert log_proba.shape == torch.Size([self.n_learnable_neurons]), \
            'Wrong log_probability shape, got: ' + str(log_proba.shape) + ', expected: ' + str(torch.Size([self.n_learnable_neurons]))

        ### Compute gradients
        if self.training:
            self.compute_gradients(self.spiking_history[self.learnable_neurons, :, -1], self.potential, ff_trace, fb_trace)

        return log_proba



    ### Initializers
    def initialize_ff_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / ((torch.sum(topology[:, i]) + torch.sum(topology[i, :])) * self.n_basis_feedforward) for i in range(self.n_learnable_neurons)]).flatten()
            std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, self.alphabet_size, self.n_neurons, self.alphabet_size, self.n_basis_feedforward)
            assert std.shape == self.ff_weights_shape
            self.feedforward_weights = (torch.normal(gain * std, std).to(self.device) * self.feedforward_mask)

        elif howto == 'uniform':
            self.feedforward_weights = (gain * (torch.rand(self.ff_weights_shape) * 2 - 1).to(self.device) * self.feedforward_mask)

        self.ff_grad = torch.zeros(self.ff_weights_shape).to(self.device)

    def initialize_fb_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / ((torch.sum(topology[:, i]) + torch.sum(topology[i, :])) * self.n_basis_feedback) for i in range(self.n_learnable_neurons)]).flatten()
            std = std.unsqueeze(1).unsqueeze(2).repeat(1, self.alphabet_size, self.n_basis_feedback)
            assert std.shape == self.fb_weights_shape
            self.feedback_weights = (torch.normal(gain * std, std)).to(self.device)
        elif howto == 'uniform':
            self.feedback_weights = (gain * (torch.rand(self.fb_weights_shape) * 2 - 1)).to(self.device)

        self.fb_grad = torch.zeros(self.fb_weights_shape).to(self.device)

    def initialize_bias_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / (torch.sum(topology[:, i]) + torch.sum(topology[i, :])) for i in range(self.n_learnable_neurons)]).flatten()
            std = std.unsqueeze(1).repeat(1, self.alphabet_size)
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
        return torch.sum(self.feedforward_weights * ff_trace, dim=(-1, -2, -3))


    def compute_fb_potential(self, fb_trace):
        return torch.sum(self.feedback_weights * fb_trace, dim=(-1))


    def generate_spikes(self, neurons_group):
        spikes = OneHotCategorical(torch.softmax(torch.cat((torch.zeros([len(neurons_group), 1]).to(self.device),
                                                            self.potential[neurons_group - self.n_input_neurons]), dim=-1), dim=-1)).sample()
        self.spiking_history[neurons_group, :, -1] = spikes[:, 1:].to(self.device)


    def update_spiking_history(self, input_signal):
        self.spiking_history = torch.cat((self.spiking_history[:, :, - self.memory_length:],
                                          torch.zeros([self.n_neurons, self.alphabet_size, 1]).to(self.device)), dim=-1)
        self.spiking_history[self.visible_neurons, :, -1] = input_signal

        if self.n_hidden_neurons > 0:
            self.generate_spikes(self.hidden_neurons)
        if not self.training:
            self.generate_spikes(self.output_neurons)


    def compute_gradients(self, spikes, potential, feedforward_trace, feedback_trace):
        self.bias_grad = spikes - custom_softmax(potential, 1, -1)
        assert self.bias_grad.shape == self.bias.shape, "Wrong bias gradient shape"

        self.ff_grad = feedforward_trace.unsqueeze(0).unsqueeze(0).repeat(self.n_learnable_neurons, self.alphabet_size, 1, 1, 1) \
                       * self.bias_grad.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, self.n_neurons, self.alphabet_size, self.n_basis_feedforward) \
                       * self.feedforward_mask
        assert self.ff_grad.shape == self.feedforward_weights.shape, "Wrong feedforward weights gradient shape"

        self.fb_grad = feedback_trace * self.bias_grad.unsqueeze(2).repeat(1, 1, self.n_basis_feedback)
        assert self.fb_grad.shape == self.feedback_weights.shape, "Wrong feedback weights gradient shape"
