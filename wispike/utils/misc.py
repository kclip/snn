import torch
import numpy as np
import pyldpc
from bitstring import BitArray


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


def channel_coding_decoding(args, message):
    # Transmit through channel
    message_shape = message.shape

    to_send = np.zeros([args.k])
    to_send[:np.prod(message_shape)] = message.flatten()

    coded_quantized = pyldpc.encode(args.G, to_send, args.snr)
    received = pyldpc.decode(args.H, coded_quantized, args.snr, args.maxiter)

    decoded = received[:np.prod(message_shape)]
    decoded = decoded.reshape(*message_shape)

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
        data_reconstructed = frames.reshape([-1, 676]).transpose(1, 0)

    return data_reconstructed


def get_intermediate_dims(vqvae, args, dataset):
    example_frame = example_to_framed(dataset.root.train.data[0], args)[0].unsqueeze(0)
    args.frame_shape = example_frame.shape

    example_quantized, example_encodings = vqvae.encode(example_frame)
    encodings_dim = example_encodings.data.numpy().shape
    quantized_dim = example_quantized.data.clone().permute(0, 2, 3, 1).contiguous().shape

    return quantized_dim, encodings_dim


def byte2bytearr(byte):
    bitstrlist = list(BitArray(hex=hex(byte)).bin)
    res = np.array([0 if i == '0' else 1 for i in bitstrlist], dtype=bool)
    if len(res) != 8:
        return np.hstack((np.zeros([4], dtype=bool), res))
    else:
        return res

def bytearr2uint8(bitarray):
    return np.uint8(int(''.join([str(int(i)) for i in list(bitarray)]), 2))


def binarr2bytearr(array):
    if (len(array) % 8) != 0:
        padding = 8 - (len(array) % 8)
        return np.hstack((array, np.zeros([padding], dtype=bool))).reshape([-1, 8])
    return array.reshape([-1, 8])


def binarr2bytes(sample):
    sample_as_bytearr = binarr2bytearr(sample)
    sample_as_bytes = np.array([], dtype=np.uint8)

    for byte in sample_as_bytearr:
        sample_as_bytes = np.hstack((sample_as_bytes, bytearr2uint8(byte)))

    return sample_as_bytes.tobytes()


def bytes2binarr(bytes_):
    res = np.array([], dtype=bool)
    for byte in bytes_:
        res = np.hstack((res, byte2bytearr(byte)))
    return res

