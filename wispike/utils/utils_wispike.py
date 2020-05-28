import torch
from binary_snn.utils_binary import misc


def get_acc(encoder, decoder, dataset, test_indices, snr, systematic, n_outputs_enc):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoder.set_mode('test')
    encoder.reset_internal_state()

    decoder.set_mode('test')
    decoder.reset_internal_state()

    S_prime = dataset.root.test.label[:].shape[-1]
    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, S_prime])

    hidden_hist = torch.zeros([encoder.n_hidden_neurons + decoder.n_hidden_neurons, S_prime])

    for j, sample_idx in enumerate(test_indices):
        misc.refractory_period(encoder)
        misc.refractory_period(decoder)
        sample_enc = torch.FloatTensor(dataset.root.test.data[sample_idx]).to(encoder.device)

        for s in range(S_prime):
            _ = encoder(sample_enc[:, s])

            if systematic:
                decoder_input = channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1])), decoder.device, snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1], decoder.device, snr)

            _ = decoder(decoder_input)
            outputs[j, :, s] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, s] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, s] = decoder.spiking_history[decoder.hidden_neurons, -1]


    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(torch.FloatTensor(dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc

def get_acc_per_samples_sent(encoder, decoder, dataset, test_indices, snr, systematic, n_outputs_enc):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoder.set_mode('test')
    encoder.reset_internal_state()

    decoder.set_mode('test')
    decoder.reset_internal_state()

    S_prime = dataset.root.test.label[:].shape[-1]
    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, S_prime])

    hidden_hist = torch.zeros([encoder.n_hidden_neurons + decoder.n_hidden_neurons, S_prime])

    for j, sample_idx in enumerate(test_indices):
        misc.refractory_period(encoder)
        misc.refractory_period(decoder)
        sample_enc = torch.FloatTensor(dataset.root.test.data[sample_idx]).to(encoder.device)

        for s in range(S_prime):
            _ = encoder(sample_enc[:, s])

            if systematic:
                decoder_input = channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1])), decoder.device, snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1], decoder.device, snr)

            _ = decoder(decoder_input)
            outputs[j, :, s] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, s] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, s] = decoder.spiking_history[decoder.hidden_neurons, -1]

    true_classes = torch.max(torch.sum(torch.FloatTensor(dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices
    accs = torch.zeros([len(test_indices)], dtype=torch.float)

    for s in range(1, S_prime):
        predictions = torch.max(torch.sum(outputs[:, :, :s], dim=-1), dim=-1).indices
        acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
        accs[s] = acc

    return acc



def channel(signal, device, snr_db):
    sig_avg_db = 10 * torch.log10(torch.mean(signal))
    noise_db = sig_avg_db - snr_db
    noise = 10 ** (noise_db / 10)
    noise = torch.normal(0, torch.ones(signal.shape) * noise)

    channel_output = signal + noise.to(device)
    return channel_output.round()

# def channel_coding_decoding(self, source_encodings, snr):
#     # Transmit through channel
#     encodings_shape = source_encodings.data.numpy().shape
#     n_bits_total = np.prod(source_encodings.size())
#     n_blocks = n_bits_total // self.k
#     residual = n_bits_total % self.k
#
#     if residual:
#         n_blocks += 1
#
#     encodings_flattened = np.zeros(self.k * n_blocks)
#     encodings_flattened[:n_bits_total] = source_encodings.flatten().data.numpy()
#
#     coded_encodings = pyldpc.encode(self.G, encodings_flattened.reshape(self.k, n_blocks), snr)
#     received_encodings = pyldpc.decode(self.H, coded_encodings, snr)
#     decoded_encodings = received_encodings[:self.k, :]
#
#     decoded_encodings = decoded_encodings.flatten()[:np.prod(encodings_shape)]
#     decoded_encodings = torch.FloatTensor(decoded_encodings.reshape(*encodings_shape))
#
#     # print('decoded encodings ', decoded_encodings.shape)
#
#     return decoded_encodings
#
#
# ### LDPC coding
# self.d_v = d_v
# self.d_c = d_c
#
# self.H, self.G = pyldpc.make_ldpc(ldpc_codewords_length, self.d_v, self.d_c, systematic=True, sparse=True)
# _, self.k = self.G.shape
