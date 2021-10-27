from __future__ import print_function
import tables
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_
from snn.utils import filters
import math
import numpy as np

class SNNetwork(torch.nn.Module):
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, topology, synaptic_filter=filters.exponential_filter,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=0.5, initialization='uniform', weights_magnitude=0.01, device='cpu', save_path=None):

        super(SNNetwork, self).__init__()
        '''
        An SNN network is defined by its topology, i.e., the connections between the neurons. 
        Parameters:
        n_input_neurons: exogeneous (non learnable) inputs
        n_hidden_neurons: hidden learnable neurons
        n_output_neurons: output neurons (visible during training)
        topology: matrix defining the synaptic connections between neurons, of size n_learnable_neurons x n_neurons 
        topology[i, j] = 1 means that there is a feedforward synaptic connection from neuron j to neuron i
        synaptic_filter: the basis function(s) used to compute contributions from pre-synaptic neurons 
        tau_ff, n_basis_feedforward: parameters of the feedforward filter
        tau_fb, n_basis_feedback: parameters of the feedback filter
        weights_magnitude: at initialization
        initialization: glorot or uniform
        device: pytorch device
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

        self.n_learnable_neurons = n_hidden_neurons + n_output_neurons
        self.learnable_neurons = torch.cat((self.hidden_neurons, self.output_neurons))

        self.training = None


        # Sanity checks
        assert self.n_learnable_neurons == topology.shape[0], 'The topology of the network should be of shape [n_learnable_neurons, n_neurons]'
        assert self.n_neurons == topology.shape[-1], 'The topology of the network should be of shape [n_learnable_neurons, n_neurons]'
        topology[[i for i in range(self.n_learnable_neurons)], self.learnable_neurons] = 0


        self.topology = topology
        self.initialization = initialization
        self.weights_magnitude = weights_magnitude


        ### Feedforward connections
        self.ff_weights_shape = None
        self.feedforward_weights = None
        self.n_basis_feedforward = n_basis_feedforward
        self.feedforward_filter = synaptic_filter(tau_ff, self.n_basis_feedforward, mu).transpose(0, 1).to(self.device)
        self.tau_ff = tau_ff


        ### Feedback connections
        self.fb_weights_shape = None
        self.feedback_weights = None
        self.n_basis_feedback = n_basis_feedback
        self.feedback_filter = synaptic_filter(tau_fb, self.n_basis_feedback, mu).transpose(0, 1).to(self.device)
        self.tau_fb = tau_fb


        ### Bias
        self.bias_shape = None
        self.bias = None


        ### Number of timesteps to keep in synaptic memory
        self.memory_length = max(self.tau_ff, self.tau_fb)


        ### State of the network
        self.spiking_history = None
        self.potential = None

        ### Gradients
        self.ff_grad = None
        self.fb_grad = None
        self.bias_grad = None


        # Path to where the weights are saved, if None they will be saved in the current directory
        self.save_path = save_path



    ### Getters
    def get_parameters(self):
        return {'ff_weights': self.feedforward_weights, 'fb_weights': self.feedback_weights, 'bias': self.bias}

    def get_gradients(self):
        return {'ff_weights': self.ff_grad, 'fb_weights': self.fb_grad, 'bias': self.bias_grad}



    ### Setters
    def set_ff_weights(self, new_weights):
        assert new_weights.shape == self.ff_weights_shape, 'Wrong shape, got ' + str(new_weights.shape) + ', expected ' + str(self.ff_weights_shape)
        self.feedforward_weights = new_weights.to(self.device)
        return


    def set_fb_weights(self, new_weights):
        assert new_weights.shape == self.fb_weights_shape, 'Wrong shape, got ' + str(new_weights.shape) + ', expected ' + str(self.fb_weights_shape)
        self.feedback_weights = new_weights.to(self.device)
        return


    def set_bias(self, new_bias):
        assert new_bias.shape == self.bias_shape, 'Wrong shape, got ' + str(new_bias.shape) + ', expected ' + str(self.bias_shape)
        self.bias = new_bias.to(self.device)
        return


    def reset_internal_state(self):
        self.spiking_history = torch.zeros(self.spiking_history.shape).to(self.device)
        self.potential = 0
        return


    def reset_weights(self):
        self.initialize_ff_weights(self.topology, howto=self.initialization, gain=self.weights_magnitude)
        self.initialize_fb_weights(self.topology, howto=self.initialization, gain=self.weights_magnitude)
        self.initialize_bias_weights(self.topology, howto=self.initialization, gain=self.weights_magnitude)
        return


    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.training = False



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




class SNNLayer(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, synaptic_filter=filters.raised_cosine_pillow_08,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=0.5, device='cpu'):
        super(SNNLayer, self).__init__()

        self.device = device

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        ### Feedforward connections
        self.n_basis_feedforward = n_basis_feedforward
        self.feedforward_filter = synaptic_filter(tau_ff, self.n_basis_feedforward, mu).transpose(0, 1).to(self.device)
        self.feedforward_filter.requires_grad = False
        self.tau_ff = tau_ff

        ### Feedback connections
        self.n_basis_feedback = n_basis_feedback
        self.feedback_filter = synaptic_filter(tau_fb, self.n_basis_feedback, mu).transpose(0, 1).to(self.device)
        self.feedback_filter.requires_grad = False
        self.tau_fb = tau_fb

        self.ff_weights = torch.nn.parameter.Parameter(torch.Tensor(n_outputs, n_inputs, n_basis_feedforward))
        self.fb_weights = torch.nn.parameter.Parameter(torch.Tensor(n_outputs, n_basis_feedback))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(n_outputs))

        a = self.get_xavier()
        _no_grad_uniform_(self.ff_weights, -a, a)
        _no_grad_uniform_(self.fb_weights, -a, a)
        _no_grad_uniform_(self.bias, -a, a)


        self.spiking_history = torch.zeros([self.n_outputs, 2], requires_grad=True).to(self.device)

        self.potential = None

        ### Number of timesteps to keep in synaptic memory
        self.memory_length = max(self.tau_ff, self.tau_fb)


    def forward(self, input_history, target=None, no_update=False):
        ff_trace = self.compute_ff_trace(input_history)
        fb_trace = self.compute_fb_trace()

        self.potential = self.compute_ff_potential(ff_trace) + self.compute_fb_potential(fb_trace) + self.bias

        outputs = self.generate_spikes(target)
        if not no_update:
            self.spiking_history = self.update_spiking_history(outputs)

        # return logits
        return torch.sigmoid(self.potential), self.spiking_history[:, -1]

    def detach_(self):
        self.potential.detach_()
        self.spiking_history.detach_()

        self.ff_weights.detach_().requires_grad_()
        self.fb_weights.detach_().requires_grad_()
        self.bias.detach_().requires_grad_()

    def compute_ff_trace(self, input_history):
        if input_history.shape[-1] != self.feedforward_filter.shape[0]:
            return torch.matmul(input_history.flip(-1), self.feedforward_filter[:input_history.shape[-1]])
        else:
            return torch.matmul(input_history.flip(-1), self.feedforward_filter)

    def compute_ff_potential(self, ff_trace):
        return torch.sum(self.ff_weights * ff_trace, dim=(-1, -2))

    def compute_fb_trace(self):
        if self.spiking_history.shape[-1] != self.feedback_filter.shape[0]:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter[:self.spiking_history.shape[-1]])
        else:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter)

    def compute_fb_potential(self, fb_trace):
        return torch.sum(self.fb_weights * fb_trace, dim=(-1))

    def generate_spikes(self, target=None):
        if target is not None:
            return target
        else:
            try:
                outputs = torch.bernoulli(torch.sigmoid(self.potential)).to(self.device)
            except RuntimeError:
                print('Potential')
                print(self.potential)
                print('ff_weights', self.ff_weights.isnan().any())
                print('fb_weights', self.fb_weights.isnan().any())
                print('bias', self.bias.isnan().any())

            return outputs


    def update_spiking_history(self, new_spikes):
        with torch.no_grad():
            spiking_history = torch.cat((self.spiking_history[:, 1-self.memory_length:], torch.zeros([self.n_outputs, 1], requires_grad=True).to(self.device)), dim=-1)
            spiking_history[:, -1] = new_spikes

            return spiking_history


    def reset_weights(self):
        torch.nn.init.xavier_uniform_(self.fb_weights)
        torch.nn.init.xavier_uniform_(self.ff_weights)
        torch.nn.init.xavier_uniform_(self.bias)

    def get_xavier(self, gain=1.):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.ff_weights)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

        return a


class SNNLayerv2(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, batch_size, synaptic_filter=filters.raised_cosine_pillow_08,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=0.5, device='cpu'):
        super(SNNLayerv2, self).__init__()

        self.device = device

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batch_size = batch_size

        ### Feedforward connections
        self.n_basis_feedforward = n_basis_feedforward
        self.feedforward_filter = synaptic_filter(tau_ff, self.n_basis_feedforward, mu).transpose(0, 1).to(self.device)
        self.feedforward_filter.requires_grad = False
        self.tau_ff = tau_ff

        ### Feedback connections
        self.n_basis_feedback = n_basis_feedback
        self.feedback_filter = synaptic_filter(tau_fb, self.n_basis_feedback, mu)[0].to(self.device)
        self.feedback_filter.requires_grad = False
        self.tau_fb = tau_fb

        self.ff_synapses = torch.nn.ModuleList([torch.nn.Linear(n_inputs, n_outputs, bias=False) for _ in range(n_basis_feedforward)])
        # [torch.nn.init.uniform_(l.weight, -1/(n_inputs + n_outputs)**2,  1/(n_inputs + n_outputs)**2) for l in self.ff_synapses]
        # [torch.nn.init.uniform_(l.weight, -1/(n_inputs + n_outputs),  0) for l in self.ff_synapses]
        self.fb_synapse = torch.nn.Linear(n_outputs, n_outputs, bias=True)
        # torch.nn.init.uniform_(self.fb_synapse.weight, -1/(2*n_outputs)**2, 1/(2*n_outputs)**2)
        # torch.nn.init.uniform_(self.fb_synapse.bias, -1/(2*n_outputs)**2, 1/(2*n_outputs)**2)
        # torch.nn.init.uniform_(self.fb_synapse.weight, -1/(2*n_outputs), 0)
        # torch.nn.init.uniform_(self.fb_synapse.bias, -1/(2*n_outputs), 0)

        self.spiking_history = torch.zeros([self.batch_size, self.n_outputs, 2], requires_grad=True).to(self.device)

        self.potential = None

        ### Number of timesteps to keep in synaptic memory
        self.memory_length = max(self.tau_ff, self.tau_fb)



    def forward(self, input_history, target=None, no_update=False):
        ff_trace = self.compute_ff_trace(input_history)
        fb_trace = self.compute_fb_trace()

        ff_potential = self.compute_ff_potential(ff_trace)
        fb_potential = self.compute_fb_potential(fb_trace)
        self.potential = ff_potential + fb_potential

        outputs = self.generate_spikes(target)
        if not no_update:
            self.spiking_history = self.update_spiking_history(outputs)

        # return logits
        return torch.sigmoid(self.potential), self.spiking_history[:, :, -1]

    def detach_(self):
        self.potential.detach_()
        self.spiking_history.detach_()

        # [l.weight.detach_().requires_grad_() for l in self.ff_synapses]
        # self.fb_synapse.weight.detach_().requires_grad_()
        # self.fb_synapse.bias.detach_().requires_grad_()

    def compute_ff_trace(self, input_history):
        # input_history: shape = [n_batch, n_in, t]
        # feedforward filter: shape = [tau_ff, n_basis_ff]
        # res: shape = [[n_batch, n_in] * n_basis_ff]
        if input_history.shape[-1] != self.feedforward_filter.shape[0]:
            return [torch.matmul(input_history.flip(-1), self.feedforward_filter[:input_history.shape[-1], i]) for i in range(self.n_basis_feedforward)]
        else:
            return [torch.matmul(input_history.flip(-1), self.feedforward_filter[:input_history.shape[-1], i]) for i in range(self.n_basis_feedforward)]

    def compute_ff_potential(self, ff_trace):
        # ff_trace: shape = [[n_batch, n_in] * n_basis_ff]
        # ff_synapses: shape = [[n_out, n_in] * n_basis_ff]

        return torch.cat([self.ff_synapses[i](ff_trace[i]).unsqueeze(2) for i in range(self.n_basis_feedforward)], dim=-1).sum(dim=-1)

    def compute_fb_trace(self):
        # input_history: shape = [n_batch, n_out, t]
        # feedforward filter: shape = [tau_fb, 1]
        # res: shape = [n_batch, n_in]
        if self.spiking_history.shape[-1] != self.feedback_filter.shape[0]:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter[:self.spiking_history.shape[-1]])
        else:
            return torch.matmul(self.spiking_history.flip(-1), self.feedback_filter)

    def compute_fb_potential(self, fb_trace):
        return self.fb_synapse(fb_trace)


    def generate_spikes(self, target=None):
        if target is not None:
            return target
        else:
            try:
                outputs = torch.bernoulli(torch.sigmoid(self.potential)).to(self.device)
            except RuntimeError:
                print('Potential')
                print(self.potential)
                print('ff_weights', self.ff_weights.isnan().any())
                print('fb_weights', self.fb_weights.isnan().any())
                print('bias', self.bias.isnan().any())

            return outputs


    def update_spiking_history(self, new_spikes):
        with torch.no_grad():
            spiking_history = torch.cat((self.spiking_history[:, :, 1-self.memory_length:],
                                         torch.zeros([self.batch_size, self.n_outputs, 1], requires_grad=True).to(self.device)), dim=-1)
            spiking_history[:, :, -1] = new_spikes

            return spiking_history


    def get_xavier(self, gain=1.):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.fb_weights)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

        return a
