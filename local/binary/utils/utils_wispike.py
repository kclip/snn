import torch
from data_preprocessing import misc


def get_acc(encoder, decoder, dataset, test_indices, noise_level):
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
            decoder_input = channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-encoder.n_output_neurons:], -1])), decoder.device, noise_level)
            # decoder_input = torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1]))

            _ = decoder(decoder_input)
            outputs[j, :, s] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, s] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, s] = decoder.spiking_history[decoder.hidden_neurons, -1]


        # print(torch.sum(hidden_hist[:encoder.n_hidden_neurons], dim=-1))
        # print(torch.sum(hidden_hist[encoder.n_hidden_neurons:], dim=-1))
        # print(torch.sum(outputs[j, :], dim=-1), torch.max(torch.sum(torch.FloatTensor(dataset.root.test.label[:][sample_idx]), dim=-1), dim=-1).indices)
        # print('//////////////////////////////////////////////////////////')

    predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
    true_classes = torch.max(torch.sum(torch.FloatTensor(dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc

def channel(channel_input, device, noise_level):
    channel_output = channel_input + torch.normal(0., torch.ones(channel_input.shape) * noise_level).to(device)
    return channel_output.round()

