import torch
from binary_snn.utils_binary import misc
import numpy as np
import pyldpc


def get_acc(encoder, decoder, args, test_indices, n_outputs_enc):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    encoder.set_mode('test')
    encoder.reset_internal_state()

    decoder.set_mode('test')
    decoder.reset_internal_state()

    S_prime = args.dataset.root.test.label[:].shape[-1]
    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, S_prime])

    hidden_hist = torch.zeros([encoder.n_hidden_neurons + decoder.n_hidden_neurons, S_prime])

    for j, sample_idx in enumerate(test_indices):
        misc.refractory_period(encoder)
        misc.refractory_period(decoder)
        sample_enc = torch.FloatTensor(args.dataset.root.test.data[sample_idx]).to(encoder.device)

        for s in range(S_prime):
            _ = encoder(sample_enc[:, s])

            if args.systematic:
                decoder_input = channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1])), decoder.device, args.snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1], decoder.device, args.snr)

            _ = decoder(decoder_input)
            outputs[j, :, s] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, s] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, s] = decoder.spiking_history[decoder.hidden_neurons, -1]


    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices
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
    sigma_noise = 10 ** (noise_db / 10)

    noise = torch.normal(0, torch.ones(signal.shape)) * sigma_noise
    channel_output = signal + noise.to(device)

    channel_output[channel_output >= 0.5] = 1
    channel_output[channel_output < 0.5] = 0
    return channel_output

