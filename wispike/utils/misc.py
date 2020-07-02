import torch
import numpy as np
import pyldpc


def binarize(signal):
    signal[signal >= 0.5] = 1.
    signal[signal < 0.5] = 0.
    return signal


def channel(signal, device, snr_db):
    sig_avg_db = 10 * torch.log10(torch.mean(signal))
    noise_db = sig_avg_db - snr_db
    sigma_noise = 10 ** (noise_db / 10)

    noise = torch.normal(0, torch.ones(signal.shape) * sigma_noise)
    channel_output = signal + noise.to(device)

    channel_output = binarize(channel_output)
    return channel_output


def channel_coding_decoding(args, encodings):
    # Transmit through channel
    encodings_shape = encodings.data.numpy().shape

    to_send = np.zeros([args.k])
    to_send[:np.prod(encodings_shape)] = encodings.clone().data.flatten()

    coded_quantized = pyldpc.encode(args.G, to_send, args.snr)
    received = pyldpc.decode(args.H, coded_quantized, args.snr, args.maxiter)

    decoded = received[:np.prod(encodings_shape)]
    decoded = torch.FloatTensor(decoded.reshape(*encodings_shape))

    return decoded


def example_to_framed(example, args):
    if args.residual:
        frames = torch.zeros([args.n_frames, 80 // (args.n_frames - 1), 26, 26])
        frames[:-1] = example[:, :(args.n_frames - 1) * (80 // (args.n_frames - 1))].transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape(frames[:-1].shape)
        frames[-1, :args.residual] = example[:, -args.residual:].transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([args.residual, 26, 26])
    else:
        frames = torch.FloatTensor(example).transpose(1, 0).unsqueeze(0).unsqueeze(3).reshape([args.n_frames, 80 // args.n_frames, 26, 26])

    return frames


def framed_to_example(frames, args):
    if args.residual:
        data_reconstructed = torch.zeros([676, 80])
        data_reconstructed[:, :(args.n_frames - 1) * (80 // (args.n_frames - 1))] \
            = frames[:-1].reshape([(args.n_frames - 1) * (80 // (args.n_frames - 1)), -1]).transpose(1, 0)
        data_reconstructed[:, -args.residual:] = frames[-1, :args.residual].reshape([args.residual, -1]).transpose(1, 0)
    else:
        data_reconstructed = frames.reshape([80, -1]).transpose(1, 0)

    return data_reconstructed

