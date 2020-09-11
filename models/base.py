from __future__ import print_function
import tables
import torch

from utils import filters


class SNNetwork(torch.nn.Module):
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, topology, synaptic_filter=filters.base_filter,
                 n_basis_feedforward=1, n_basis_feedback=1, tau_ff=1, tau_fb=1, mu=1, initialization='uniform', weights_magnitude=0.01, device='cpu', save_path=None):

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

        self.visible_neurons = None
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
        if mode:
            self.visible_neurons = torch.cat((self.input_neurons, self.output_neurons))
        else:
            self.visible_neurons = self.input_neurons
        self.training = mode

    def eval(self):
        self.visible_neurons = self.input_neurons
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


