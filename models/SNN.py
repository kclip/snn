from __future__ import print_function
import torch
import numpy as np
import utils.filters as filters
import tables
import os


class SNNetwork(torch.nn.Module):
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, topology, synaptic_filter=filters.base_feedforward_filter,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=1, initialization='uniform', weights_magnitude=0.01, device='cpu', save_path=None):

        super(SNNetwork, self).__init__()
        '''
        An SNN network is defined by its topology, i.e., the connections between the neurons. 
        A forward pass in the network consists of transmitting information from the input neurons to the rest of the network, starting with the input neurons.
        The behavior of visible neurons is given by the input during the pass. 
        Hidden neurons produce spikes following a Bernoulli distribution parametrized by the sigmoid of their membrane potential.  
        Parameters: 
        topology: matrix defining the synaptic connections between neurons, of size n_learnable_neurons x n_neurons 
        topology[i, j] = 1 means that there is a feedforward synaptic connection from neuron j to neuron i
        visible neurons: neurons for which the behavior is determined by the input signal
        feedforward_filter: the basis function(s) used to compute contributions from pre-synaptic neurons
        feedback_filter: the basis function(s) used to compute contributions from the history 
        tau_ff, n_basis_feedforward: parameters of the feedforward filter
        tau_fb, n_basis_feedback: parameters of the feedback filter
        weights_magnitude: the weights are initialized following an uniform distribution between [-weights_magnitude, +weights_magnitude]
        minibatch_size
        '''

        self.device = device

        ### Network parameters
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.n_neurons = n_input_neurons + n_hidden_neurons + n_output_neurons
        self.weights_magnitude = weights_magnitude


        ### Neurons indices
        self.input_neurons = torch.LongTensor([i for i in range(self.n_input_neurons)])
        self.hidden_neurons = torch.LongTensor([self.n_input_neurons + i for i in range(self.n_hidden_neurons)])
        self.output_neurons = torch.LongTensor([self.n_input_neurons + self.n_hidden_neurons + i for i in range(self.n_output_neurons)])


        # In supervised mode, we avoid computations with unnecessary large matrices
        self.n_learnable_neurons = n_hidden_neurons + n_output_neurons
        self.learnable_neurons = torch.cat((self.hidden_neurons, self.output_neurons))


        # Set train mode, to avoid spurious computations of gradients during test
        self.mode = mode
        if self.mode == 'train':
            self.visible_neurons = torch.cat((self.input_neurons, self.output_neurons))
        elif self.mode == 'test':
            self.visible_neurons = self.input_neurons
        else:
            print('Mode should be one of "train" or "test"')
            raise AttributeError
        #todo change this for train/eval in pytorch
        # todo harmonize behaviors between getters for gradients / parameters


        # Sanity checks
        assert self.n_learnable_neurons == topology.shape[0], 'The topology of the network should be of shape [n_learnable_neurons, n_neurons]'
        assert self.n_neurons == topology.shape[-1], 'The topology of the network should be of shape [n_learnable_neurons, n_neurons]'
        topology[[i for i in range(self.n_learnable_neurons)], self.learnable_neurons] = 0


        ### Feedforward weights
        self.n_basis_feedforward = n_basis_feedforward
        # Creating the feedforward weights according to the topology.
        # Feedforward weights are a tensor of size [n_learnable_neurons, n_neurons, n_basis_feedforward] for which the block-diagonal elements are 0,
        # and otherwise feedforward_weights[i, j, :] ~ Unif[-weights_magnitude, +weights_magnitude] if topology[i, j] = 1
        self.feedforward_mask = torch.tensor(np.kron(topology, np.ones([self.n_basis_feedforward]))
                                             .reshape([self.n_learnable_neurons, self.n_neurons, self.n_basis_feedforward]), dtype=torch.float).to(self.device)
        self.feedforward_weights = None
        self.initialize_ff_weights(topology, howto=initialization, gain=weights_magnitude)
        self.feedforward_filter = feedforward_filter(tau_ff, self.n_basis_feedforward, mu).to(self.device)
        self.tau_ff = tau_ff


        ### Feedback weights
        self.n_basis_feedback = n_basis_feedback
        # Creating the feedback weights.
        # Feedback weights are a tensor of size [n_neurons, n_basis_feedback],
        # for which learnable elements are initialized as ~ Unif[-weights_magnitude, +weights_magnitude],
        self.feedback_weights = None
        self.initialize_fb_weights(topology, howto=initialization, gain=weights_magnitude)
        self.feedback_filter = feedback_filter(tau_fb, self.n_basis_feedback, mu).to(self.device)
        self.tau_fb = tau_fb


        ### Bias
        self.bias = None
        self.initialize_bias_weights(topology, howto=initialization, gain=weights_magnitude)

        # Number of timesteps to keep in memory
        self.memory_length = max(self.tau_ff, self.tau_fb)

        ### State of the network
        self.spiking_history = torch.zeros([self.n_neurons, 1]).to(self.device)
        self.potential = None

        ### Gradients
        self.gradients = {'ff_weights': torch.zeros(self.feedforward_weights.shape).to(self.device),
                          'fb_weights': torch.zeros(self.feedback_weights.shape).to(self.device),
                          'bias': torch.zeros(self.bias.shape).to(self.device)}

        # Path to where the weights are saved, if None they will be saved in the current directory
        self.save_path = save_path


    def forward(self, input_signal):

        assert self.n_neurons == (len(self.input_neurons) + len(self.hidden_neurons) + len(self.output_neurons)), "The numbers of neurons don't match"
        assert self.n_neurons == (len(self.learnable_neurons) + len(self.input_neurons)), "The numbers of neurons don't match"


        ### Compute potential
        self.potential = self.compute_ff_potential() + self.compute_fb_potential() + self.bias

        ### Update spiking history
        self.spiking_history = self.update_spiking_history(input_signal)

        ### Compute log-probabilities
        # noinspection PyTypeChecker
        log_proba = self.spiking_history[self.learnable_neurons, -1] * torch.log(1e-07 + torch.sigmoid(self.potential)) \
                    + (1 - self.spiking_history[self.learnable_neurons, -1]) * torch.log(1. + 1e-07 - torch.sigmoid(self.potential))  # We add 1e-07 for numerical stability of the log

        assert log_proba.shape == torch.Size([self.n_learnable_neurons]), \
            'Wrong log_probability shape, got: ' + str(log_proba.shape) + ', expected: ' + str(torch.Size([self.n_learnable_neurons]))


        ### Compute gradients
        if self.mode != 'test':
            self.gradients = self.compute_gradients(self.spiking_history[self.learnable_neurons, -1],
                                                    self.potential,
                                                    self.compute_ff_trace(self.spiking_history[:, :-1]),
                                                    self.compute_fb_trace(self.spiking_history[:, :-1]))


        return log_proba


    ### Getters
    def get_parameters(self):
        return {'ff_weights': self.feedforward_weights, 'fb_weights': self.feedback_weights, 'bias': self.bias}


    def get_history(self):
        return self.spiking_history


    def initialize_ff_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / ((torch.sum(topology[:, i]) + torch.sum(topology[i, :])) * self.n_basis_feedforward) for i in range(self.n_learnable_neurons)]).flatten()
            std = std.unsqueeze(1).unsqueeze(2).repeat(1, self.n_neurons, self.n_basis_feedforward)
            assert std.shape == torch.Size([self.n_learnable_neurons, self.n_neurons, self.n_basis_feedforward])
            self.feedforward_weights = (torch.normal(gain * std, std).to(self.device) * self.feedforward_mask)
        elif howto == 'uniform':
            self.feedforward_weights = (gain * (torch.rand(torch.Size([self.n_learnable_neurons, self.n_neurons, self.n_basis_feedforward])) * 2 - 1).to(self.device) * self.feedforward_mask)

    def initialize_fb_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.) / (torch.sum(topology[:, i]) + torch.sum(topology[i, :]))) for i in range(self.n_learnable_neurons)]).flatten()
            std = std.unsqueeze(1).repeat(1, self.n_basis_feedback)
            assert std.shape == torch.Size([self.n_learnable_neurons, self.n_basis_feedback])
            self.feedback_weights = (torch.normal(gain * std, std)).to(self.device)
        elif howto == 'uniform':
            self.feedback_weights = (gain * (torch.rand(torch.Size([self.n_learnable_neurons, self.n_basis_feedback])) * 2 - 1)).to(self.device)


    def initialize_bias_weights(self, topology, howto='glorot', gain=0.):
        if howto == 'glorot':
            std = torch.tensor([torch.sqrt(torch.tensor(2.)) / (torch.sum(topology[:, i]) + torch.sum(topology[i, :])) for i in range(self.n_learnable_neurons)]).flatten()
            assert std.shape == torch.Size([self.n_learnable_neurons])
            self.bias = (torch.normal(gain * std, std)).to(self.device)
        elif howto == 'uniform':
            self.bias = (gain * (torch.rand([self.n_learnable_neurons]) * 2 - 1)).to(self.device)



    ### Setters
    def reset_internal_state(self):
        self.spiking_history = torch.zeros([self.n_neurons, 1]).to(self.device)
        self.potential = 0
        return


    def reset_weights(self):
        self.feedforward_weights = self.weights_magnitude * (torch.rand(self.feedforward_weights.shape) * 2 - 1) * self.feedforward_mask
        self.feedback_weights = self.weights_magnitude * (torch.rand(self.feedback_weights.shape) * 2 - 1)
        self.bias = self.weights_magnitude * (torch.rand(self.bias.shape) * 2 - 1)
        return


    def set_ff_weights(self, new_weights):
        assert new_weights.shape == self.feedforward_weights.shape, 'Wrong shape, got ' + str(new_weights.shape) + ', expected ' + str(self.feedforward_weights.shape)
        self.feedforward_weights = new_weights.to(self.device)
        return


    def set_fb_weights(self, new_weights):
        assert new_weights.shape == self.feedback_weights.shape, 'Wrong shape, got ' + str(new_weights.shape) + ', expected ' + str(self.feedback_weights.shape)
        self.feedback_weights = new_weights.to(self.device)
        return


    def set_bias(self, new_bias):
        assert new_bias.shape == self.bias.shape, 'Wrong shape, got ' + str(new_bias.shape) + ', expected ' + str(self.bias.shape)
        self.bias = new_bias.to(self.device)
        return


    def set_mode(self, mode):
        if mode == 'train':
            self.visible_neurons = torch.cat((self.input_neurons, self.output_neurons))
            self.mode = 'train'
        elif mode == 'test':
            self.visible_neurons = self.input_neurons
            self.mode = 'test'
        else:
            print('Mode should be one of "train" or "test"')
            raise AttributeError
        return


    ### Computations
    def compute_ff_trace(self, spikes):
        return torch.matmul(spikes.flip(-1), self.feedforward_filter[:, :spikes.shape[-1]].transpose(0, 1))


    def compute_fb_trace(self, spikes):
        return torch.matmul(spikes.flip(-1), self.feedback_filter[:, :spikes.shape[-1]].transpose(0, 1))


    def compute_ff_potential(self):
        return torch.sum(self.feedforward_weights * self.compute_ff_trace(self.spiking_history) * self.feedforward_mask, dim=(-1, -2))


    def compute_fb_potential(self):
        return torch.sum(self.feedback_weights * self.compute_fb_trace(self.spiking_history)[self.learnable_neurons, :], dim=(-1))


    def generate_spikes(self, spiking_history, neurons_group):
        spiking_history[neurons_group, -1] = torch.bernoulli(torch.sigmoid(self.potential[neurons_group - self.n_input_neurons])).to(self.device)
        return spiking_history


    def update_spiking_history(self, input_signal):
        spiking_history = torch.cat((self.spiking_history[:, - self.memory_length + 1:],
                                          torch.zeros([self.n_neurons, 1]).to(self.device)), dim=-1)
        spiking_history[self.visible_neurons, -1] = input_signal

        if self.n_hidden_neurons > 0:
            spiking_history = self.generate_spikes(spiking_history, self.hidden_neurons)
        if self.mode == 'test':
            spiking_history = self.generate_spikes(spiking_history, self.output_neurons)

        return spiking_history


    def compute_gradients(self, spikes, potential, feedforward_trace, feedback_trace):
        bias_gradient = spikes - torch.sigmoid(potential)
        assert bias_gradient.shape == self.bias.shape, "Wrong bias gradient shape"

        ff_gradient = feedforward_trace.unsqueeze(0).repeat(self.n_learnable_neurons, 1, 1) \
                      * bias_gradient.unsqueeze(1).repeat(1, self.n_neurons).unsqueeze(2).repeat(1, 1, self.n_basis_feedforward) \
                      * self.feedforward_mask
        assert ff_gradient.shape == self.feedforward_weights.shape, "Wrong feedforward weights gradient shape"

        fb_gradient = feedback_trace[self.learnable_neurons, :] * bias_gradient.unsqueeze(1).repeat(1, self.n_basis_feedback)
        assert fb_gradient.shape == self.feedback_weights.shape, "Wrong feedback weights gradient shape"

        return {'ff_weights': ff_gradient, 'fb_weights': fb_gradient, 'bias': bias_gradient}



    ### Misc
    def save(self, path=None):
        if path is not None:
            save_path = path
        elif path is None and self.save_path is not None:
            save_path = self.save_path
        else:
            raise FileNotFoundError

        hdf5_file = tables.open_file(save_path, mode='w')
        weights_ff = hdf5_file.create_array(hdf5_file.root, 'ff_weights', self.feedforward_weights.data.cpu().numpy())
        weights_fb = hdf5_file.create_array(hdf5_file.root, 'fb_weights', self.feedback_weights.data.cpu().numpy())
        bias = hdf5_file.create_array(hdf5_file.root, 'bias', self.bias.data.cpu().numpy())
        hdf5_file.close()
        return


    def import_weights(self, path):
        hdf5_file = tables.open_file(path, mode='r')
        self.set_ff_weights(torch.tensor((hdf5_file.root['ff_weights'][:])))
        self.set_fb_weights(torch.tensor((hdf5_file.root['fb_weights'][:])))
        self.set_bias(torch.tensor((hdf5_file.root['bias'][:])))
        hdf5_file.close()
        return


