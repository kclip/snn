import torch
from models.SNN import SNNetwork
from wispike.models.mlp import MLP
import utils.utils_snn as misc_snn
from wispike.utils.misc import channel_coding_decoding, channel, framed_to_example, example_to_framed, binarize
from data_preprocessing.load_data import *


def classify(classifier, example, args, howto='final'):
    if isinstance(classifier, SNNetwork):
        if example.shape != torch.Size([676, 80]):
            example = framed_to_example(example, args)
        # SNNs only accept binary inputs
        example = binarize(example)

        predictions_final, predictions_pf = classify_snn(classifier, example, args, howto)

    elif isinstance(classifier, MLP):
        predictions_final, predictions_pf = classify_mlp(classifier, example, args, howto)

    return predictions_final, predictions_pf


def classify_snn(network, example, args, howto='final'):
    network.set_mode('test')
    network.reset_internal_state()

    T = args.dataset.root.test.label[:].shape[-1]
    outputs = torch.zeros([network.n_output_neurons, T])

    for t in range(T):
        _ = network(example[:, t])
        outputs[:, t] = network.spiking_history[network.output_neurons, -1]

    if howto == 'final':
        predictions_final = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
        return predictions_final, 0

    elif howto == 'per_frame':
        predictions_pf = torch.zeros([T])

        for t in range(T):
            predictions_pf[t] = torch.max(torch.sum(outputs[:, :t], dim=-1), dim=-1).indices

        return 0, predictions_pf

    elif howto == 'both':
        predictions_final = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
        predictions_pf = torch.zeros([T])

        for t in range(T):
            predictions_pf[t] = torch.max(torch.sum(outputs[:, :t], dim=-1), dim=-1).indices
        return predictions_final, predictions_pf

    else:
        raise NotImplementedError


def classify_mlp(network, example, args, howto='final'):
    if howto == 'final':
        inputs = framed_to_example(example, args).flatten()
        output = network(inputs)
        predictions_final = torch.argmax(output)
        return predictions_final, 0

    elif howto == 'per_frame':
        example_padded = torch.zeros(example.shape)
        predictions_pf = torch.zeros([args.n_frames])

        for i in range(example.shape[0]):
            example_padded[i] = example[i]
            inputs = framed_to_example(example_padded, args).flatten()
            output = network(inputs)
            predictions_pf[i] = torch.argmax(output)
        return 0, predictions_pf

    elif howto == 'both':
        inputs = framed_to_example(example, args).flatten()
        output = network(inputs)
        predictions_final = torch.argmax(output)
        example_padded = torch.zeros(example.shape)
        predictions_pf = torch.zeros([args.n_frames])

        for i in range(example.shape[0]):
            example_padded[i] = example[i]
            inputs = framed_to_example(example_padded, args).flatten()
            output = network(inputs)
            predictions_pf[i] = torch.argmax(output)

        return predictions_final, predictions_pf

    else:
        raise NotImplementedError

    return predictions


def get_acc_classifier(classifier, vqvae, args, indices, howto='final'):
    vqvae.eval()

    predictions_final = torch.zeros([len(indices)], dtype=torch.long)
    if args.classifier == 'snn':
        predictions_pf = torch.zeros([len(indices), 80], dtype=torch.long)
    else:
        predictions_pf = torch.zeros([len(indices), args.n_frames], dtype=torch.long)

    for i, idx in enumerate(indices):
        data = example_to_framed(args.dataset.root.test.data[idx, :, :], args)
        data_reconstructed = torch.zeros(data.shape)

        for j in range(args.n_frames):
            frame = data[j].unsqueeze(0)

            with torch.autograd.no_grad():
                _, encodings = vqvae.encode(frame)

                encodings_decoded = torch.FloatTensor(channel_coding_decoding(args, encodings))

                data_reconstructed[j] = vqvae.decode(encodings_decoded, args.quantized_dim)

        predictions_final[i], predictions_pf[i] = classify(classifier, data_reconstructed, args, howto)

    true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][indices]), dim=-1), dim=-1).indices

    if howto == 'final':
        accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))

        return accs_final, None

    elif howto == 'per_frame':
        accs_pf = torch.zeros([args.n_frames], dtype=torch.float)
        for i in range(args.n_frames):
            acc = float(torch.sum(predictions_pf[:, i] == true_classes, dtype=torch.float) / len(predictions_pf))
            accs_pf[i] = acc

        return None, accs_pf

    elif howto == 'both':
        accs_final = float(torch.sum(predictions_final == true_classes, dtype=torch.float) / len(predictions_final))
        accs_pf = torch.zeros(predictions_pf.shape, dtype=torch.float)

        if args.classifier == 'snn':
            T = 80
        else:
            T = args.n_frames

        for i in range(T):
            acc = float(torch.sum(predictions_pf[:, i] == true_classes, dtype=torch.float) / len(predictions_pf))
            accs_pf[i] = acc

        return accs_final, accs_pf


def get_acc_wispike(encoder, decoder, n_output_enc, hdf5_group, test_indices, T, n_classes, input_shape, dt, x_max, polarity, systematic, snr, howto='final'):
    encoder.eval()
    encoder.reset_internal_state()

    decoder.eval()
    decoder.reset_internal_state()

    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, T])
    loss = 0

    true_classes = torch.LongTensor(hdf5_group.labels[test_indices, 0])
    hidden_hist = torch.zeros([encoder.n_hidden_neurons + decoder.n_hidden_neurons, T])

    for j, idx in enumerate(test_indices):
        misc_snn.refractory_period(encoder)
        misc_snn.refractory_period(decoder)

        sample_enc, _ = get_example(hdf5_group, idx, T, n_classes, input_shape, dt, x_max, polarity)
        sample_enc = sample_enc.to(encoder.device)


        for t in range(T):
            _ = encoder(sample_enc[:, t])

            if systematic:
                decoder_input = channel(torch.cat((sample_enc[:, t], encoder.spiking_history[encoder.hidden_neurons[-n_output_enc:], -1])), decoder.device, snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-n_output_enc:], -1], decoder.device, snr)

            _ = decoder(decoder_input)
            outputs[j, :, t] = decoder.spiking_history[decoder.output_neurons, -1]

            hidden_hist[:encoder.n_hidden_neurons, t] = encoder.spiking_history[encoder.hidden_neurons, -1]
            hidden_hist[encoder.n_hidden_neurons:, t] = decoder.spiking_history[decoder.hidden_neurons, -1]


    if howto == 'final':
        predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
        accs_final = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

        return accs_final, None

    elif howto == 'per_frame':
        accs_pf = torch.zeros([T], dtype=torch.float)

        for t in range(1, T):
            predictions = torch.sum(outputs[:, :, :t], dim=-1).argmax(-1)
            acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
            accs_pf[t] = acc

        return None, accs_pf

    elif howto == 'both':
        predictions = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices
        accs_final = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

        accs_pf = torch.zeros([T], dtype=torch.float)
        for t in range(1, T):
            predictions = torch.sum(outputs[:, :, :t], dim=-1).argmax(-1)
            acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
            accs_pf[t] = acc

        return accs_final, accs_pf


def get_acc_jscc(encoder, decoder, args, test_indices, n_outputs_enc):
    encoder.set_mode('test')
    encoder.reset_internal_state()

    decoder.set_mode('test')
    decoder.reset_internal_state()

    T = args.dataset.root.test.label[:].shape[-1]
    outputs = torch.zeros([len(test_indices), decoder.n_output_neurons, T])

    for j, sample_idx in enumerate(test_indices):
        misc_snn.refractory_period(encoder)
        misc_snn.refractory_period(decoder)
        sample_enc = torch.FloatTensor(args.dataset.root.test.data[sample_idx]).to(encoder.device)

        for t in range(T):
            _ = encoder(sample_enc[:, t])

            if args.systematic:
                decoder_input = channel(torch.cat((sample_enc[:, t], encoder.spiking_history[encoder.hidden_neurons[-n_outputs_enc:], -1])), decoder.device, args.snr)
            else:
                decoder_input = channel(encoder.spiking_history[encoder.hidden_neurons[-args.n_output_enc:], -1], decoder.device, args.snr)

            _ = decoder(decoder_input)
            outputs[j, :, t] = decoder.spiking_history[decoder.output_neurons, -1]

    acc = float(torch.sum(outputs[:, :, 2:] == torch.FloatTensor(args.dataset.root.test.data[:][test_indices, :, :-2]))) / outputs.numel()

    return acc
