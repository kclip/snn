import torch
from binary_snn.models.SNN import SNNetwork
from wispike.models.mlp import MLP
import binary_snn.utils_binary.misc as misc_snn
from wispike.utils.misc import channel_coding_decoding, channel, framed_to_example, example_to_framed
import numpy as np


def classify(classifier, example, args, howto='final'):
    if isinstance(classifier, SNNetwork):
        example = framed_to_example(example, args)
        predictions = classify_snn(classifier, example, args, howto)

    elif isinstance(classifier, MLP):
        predictions = classify_mlp(classifier, example, args, howto)

    return predictions


def classify_snn(network, example, args, howto='final'):
    network.set_mode('test')
    network.reset_internal_state()

    S_prime = args.dataset.root.test.label[:].shape[-1]
    outputs = torch.zeros([network.n_output_neurons, S_prime])

    for s in range(S_prime):
        _ = network(example[:, s])
        outputs[:, s] = network.spiking_history[network.output_neurons, -1]

    if howto == 'final':
        predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices

    elif howto == 'per_frame':
        predictions = torch.zeros([S_prime])
        for s in range(1, S_prime):
            predictions[s] = torch.max(torch.sum(outputs[:, :s], dim=-1), dim=-1).indices
    else:
        raise NotImplementedError

    return predictions


def classify_mlp(network, example, args, howto='final'):
    example_padded = torch.zeros(example.shape)

    if howto == 'final':
        inputs = framed_to_example(example, args).flatten()
        output = network(inputs)
        predictions = torch.argmax(output)

    elif howto == 'per_frame':
        predictions = torch.zeros([example.shape[0]])

        for i in range(example.shape[0]):
            example_padded[i] = example[i]
            inputs = framed_to_example(example_padded, args).flatten()
            output = network(inputs)
            predictions[i] = torch.argmax(output)

    else:
        raise NotImplementedError

    return predictions


def get_acc_classifier(classifier, vqvae, args, indices, howto='final'):
    vqvae.eval()

    if howto == 'final':
        predictions = torch.zeros([len(indices)], dtype=torch.long)
    elif howto == 'per_frame':
        predictions = torch.zeros([len(indices), args.n_frames], dtype=torch.long)
    else:
        raise NotImplementedError

    for i, idx in enumerate(indices[:1]):
        data = example_to_framed(args.dataset.root.test.data[idx, :, :], args)
        data_reconstructed = torch.zeros(data.shape)

        for j in range(args.n_frames):
            frame = data[j].unsqueeze(0)

            with torch.autograd.no_grad():
                _, encodings = vqvae.encode(frame)

                encodings_decoded = channel_coding_decoding(args, encodings)

                data_reconstructed[j] = vqvae.decode(encodings_decoded, args.quantized_dim)

        predictions[i] = classify(classifier, data_reconstructed, args, howto)
        data_reconstructed[data_reconstructed < 0.5] = 0. # todo
        data_reconstructed[data_reconstructed >= 0.5] = 1. # todo

        print(float(torch.sum(data == data_reconstructed)), float(torch.sum(data == data_reconstructed)) / np.prod(data.shape))  # todo
        print(data[:10, 0, 0, 0])
        print(data_reconstructed[:10, 0, 0, 0])

    true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][indices]), dim=-1), dim=-1).indices

    if howto == 'final':
        accs = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    elif howto == 'per_frame':
        accs = torch.zeros([args.n_frames], dtype=torch.float)

        for i in range(args.n_frames):
            acc = float(torch.sum(predictions[:, i] == true_classes, dtype=torch.float) / len(predictions))
            accs[i] = acc

    return accs


def get_acc_wispike(encoder, decoder, args, test_indices, n_outputs_enc, howto='final'):
    encoder.set_mode('test')
    encoder.reset_internal_state()

    decoder.set_mode('test')
    decoder.reset_internal_state()

    S_prime = args.dataset.root.test.label[:].shape[-1]
    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, S_prime])

    hidden_hist = torch.zeros([encoder.n_hidden_neurons + decoder.n_hidden_neurons, S_prime])

    for j, sample_idx in enumerate(test_indices):
        misc_snn.refractory_period(encoder)
        misc_snn.refractory_period(decoder)
        sample_enc = torch.FloatTensor(args.dataset.root.test.data[sample_idx]).to(encoder.device)

        for s in range(S_prime):
            _ = encoder(sample_enc[:, s])

            if args.systematic:
                decoder_input = channel(torch.cat((sample_enc[:, s], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1])), decoder.device, args.snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1], decoder.device, args.snr)

            _ = decoder(decoder_input)
            outputs[j, :, s] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, s] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, s] = decoder.spiking_history[decoder.hidden_neurons, -1]

    true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][test_indices]), dim=-1), dim=-1).indices

    if howto == 'final':
        predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
        accs = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    elif howto == 'per_frame':
        accs = torch.zeros([S_prime], dtype=torch.float)

        for s in range(1, S_prime):
            predictions = torch.sum(outputs[:, :, :s], dim=-1).argmax(-1)
            acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
            accs[s] = acc
    else:
        raise NotImplementedError

    return accs
