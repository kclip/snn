import torch
from binary_snn.models.SNN import SNNetwork
from wispike.models.mlp import MLP
import numpy as np
import pyldpc


def classify(classifier, sample):
    if isinstance(classifier, SNNetwork):
        S_prime = sample.shape[-1]
        outputs = torch.zeros([classifier.n_output_neurons, S_prime])

        for s in range(S_prime):
            _ = classifier(sample[:, s])
            outputs[:, s] = classifier.spiking_history[classifier.output_neurons, -1]

        prediction = torch.max(torch.sum(outputs, dim=-1), dim=-1).indices

    elif isinstance(classifier, MLP):
        sample = sample.flatten()

        output = classifier(sample)
        prediction = torch.argmax(output)


    return prediction


def channel_coding_decoding(args, quantized):
    # Transmit through channel
    quantized_shape = quantized.data.numpy().shape
    # n_bits_total = np.prod(quantized.size())

    assert np.prod(quantized.size()) == args.k  # We're sending one block
    # n_blocks = n_bits_total // k
    # residual = n_bits_total % k

    # if residual:
    #     n_blocks += 1

    # encodings_flattened = np.zeros(n_bits_total)
    # encodings_flattened[:n_bits_total] = source_encodings.flatten().data.numpy()

    coded_quantized = pyldpc.encode(args.G, quantized.data.numpy(), args.snr)
    received = pyldpc.decode(args.H, coded_quantized, args.snr)
    decoded = received[:args.k, :]

    decoded = decoded[:np.prod(quantized_shape)]
    decoded = torch.FloatTensor(decoded.reshape(*quantized_shape))

    return decoded


def test(classifier, vqvae, args, indices):
    predictions = torch.zeros([len(indices)])

    for i, sample_idx in enumerate(indices):
        data = torch.FloatTensor(args.dataset.root.test.data[sample_idx, :, :]).transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([n_frames, int(80 / n_frames), 26, 26])

        quantized = vqvae.encode(data)

        quantized = channel_coding_decoding(args, quantized, args.snr)

        sample_reconstructed = vqvae.decode(quantized)

        predictions[i] = classify(classifier, sample_reconstructed)

    true_classes = torch.max(torch.sum(torch.FloatTensor(args.dataset.root.test.label[:][indices]), dim=-1), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc
