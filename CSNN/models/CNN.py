import torch
from CSNN.models.base import SNNLayer, SNNetwork
import numpy as np
from CSNN.utils import get_output_shape
from utils import filters


class CSNN(SNNetwork):
    def __init__(self,
                 input_shape,
                 output_shape,
                 Nhid=[1],
                 kernel_size=[7],
                 stride=[1],
                 n_neurons=[128],
                 ff_filter=['base_ff_filter'],
                 fb_filter=['base_fb_filter'],
                 K=[1],
                 tau_ff=[10],
                 tau_fb=[10],
                 num_layers=1,
                 device='cpu'):

        if len(kernel_size) == 1:
            kernel_size = kernel_size * num_layers
        if len(stride) == 1:
            stride = stride * num_layers
        if len(tau_ff) == 1:
            tau_ff = tau_ff * num_layers
        if len(tau_fb) == 1:
            tau_fb = tau_fb * num_layers
        if len(K) == 1:
            K = K * num_layers
        if len(ff_filter) == 1:
            ff_filter = ff_filter * num_layers
        if len(fb_filter) == 1:
            fb_filter = fb_filter * num_layers
        if len(n_neurons) == 1:
            self.n_neurons = n_neurons * num_layers
        if Nhid is None:
            self.Nhid = Nhid = []


        super(CSNN, self).__init__()
        # Computing padding to preserve feature size
        padding = (np.array(kernel_size) - 1) // 2

        self.dropout_layers = torch.nn.ModuleList()
        self.input_shape = input_shape
        Nhid = [input_shape[0]] + Nhid
        self.num_conv_layers = num_layers

        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]


        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_layer = SNNLayer(None,
                                    n_basis_feedforward=1,
                                    feedforward_filter=None, tau_ff=1,
                                    n_basis_feedback=1, feedback_filter=None, tau_fb=1, mu=1,
                                    device=device, save_path=None)

        for i in range(self.num_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=1)
            base_layer = torch.nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = SNNLayer(base_layer,
                             n_basis_feedforward=1,
                             feedforward_filter=filters.base_feedforward_filter, tau_ff=1,
                             n_basis_feedback=1, feedback_filter=filters.base_feedback_filter, tau_fb=1, mu=1,
                             device='cpu', save_path=None)
            self.layers.append(layer)

        mlp_in = int(feature_height * feature_width * Nhid[-1])
        readout = torch.nn.Linear(mlp_in, output_shape)

        self.out_layer = readout


    def forward(self, inputs):
        log_proba = 0
        for i, layer in enumerate(self.layers):
            if i == 0:
                inputs = layer.feedforward_filter(layer.spiking_history)

            inputs = layer.feedforward_filter(layer.spiking_history)

            log_proba += layer(input)

        inputs = inputs.view(inputs.size(0), -1)
        return s_out, r_out, u_out



