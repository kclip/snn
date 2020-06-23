import torch
from utils import filters


class SNNLayer(torch.nn.Module):
    def __init__(self, layer,
                 n_basis_feedforward=1, feedforward_filter=filters.base_feedforward_filter, tau_ff=1,
                 n_basis_feedback=1, feedback_filter=filters.base_feedback_filter, tau_fb=1, mu=1,
                 device='cpu', save_path=None):

        super(SNNLayer, self).__init__()
        self.base_layer = layer

        self.n_basis_feedforward = n_basis_feedforward
        self.feedforward_filter = feedforward_filter(tau_ff, self.n_basis_feedforward, mu).to(self.device)
        self.tau_ff = tau_ff

        self.n_basis_feedback = n_basis_feedback
        self.feedback_filter = feedback_filter(tau_fb, self.n_basis_feedback, mu).to(self.device)
        self.tau_fb = tau_fb
        self.feedback_weights = torch.Parameter(0.01 * (torch.rand(torch.Size([self.n_basis_feedback])) * 2 - 1)).to(self.device)
        self.bias = torch.Parameter(0.01 * (torch.rand([1]) * 2 - 1)).to(self.device)

        self.memory_length = max(self.tau_ff, self.tau_fb)

        ### State of the network
        self.spiking_history = None
        self.potential = None

        self.device = device
        self.save_path = save_path


    def forward(self, inputs):
        ### Compute potential
        if self.base_layer is not None:
            self.potential = self.compute_ff_potential(inputs) + self.compute_fb_potential() + self.bias

        ### Update spiking history
        self.update_spiking_history(inputs)

        ### Compute log-probabilities
        if self.base_layer is not None:
            log_proba = self.spiking_history[-1] * torch.log(1e-07 + torch.sigmoid(self.potential)) \
                        + (1 - self.spiking_history[-1]) * torch.log(1. + 1e-07 - torch.sigmoid(self.potential)) # We add 1e-07 for numerical stability of the log

        return - log_proba


    def compute_ff_potential(self, inputs):
        return self.base_layer(inputs)

    def compute_fb_potential(self):
        return torch.sum(self.feedback_weights * self.compute_fb_trace(), dim=(-1))

    def compute_fb_trace(self):
        return torch.matmul(self.feedback_filter[:, :self.spiking_history.shape[-1]], self.spiking_history.flip(0))


    def update_spiking_history(self, teaching_signal):
        if teaching_signal is not None:
            if self.spiking_history is None:
                self.spiking_history = teaching_signal
            else:
                spiking_history = torch.cat((self.spiking_history[- self.memory_length + 1:].data,
                                             torch.zeros([1] + list(teaching_signal.shape)).to(self.device)), dim=-1)
                spiking_history[-1] = teaching_signal
                self.spiking_history = spiking_history
            return
        else:
            if self.spiking_history is None:
                self.spiking_history = torch.bernoulli(torch.sigmoid(self.potential)).to(self.device)
            else:
                spiking_history = torch.cat((self.spiking_history[- self.memory_length + 1:].data,
                                            torch.zeros([1] + list(self.spiking_history.shape)[1:]).to(self.device)), dim=-1)
                spiking_history[-1] = torch.bernoulli(torch.sigmoid(self.potential)).to(self.device)
                self.spiking_history = spiking_history


class SNNetwork(torch.nn.Module):
    def __init__(self):
        super(SNNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.out_layer = None

    def __len__(self):
        return len(self.LIF_layers)


    def forward(self, input):
        raise NotImplemented('')





