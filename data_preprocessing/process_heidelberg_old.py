import tables
import numpy as np
import math
from data_preprocessing.misc import one_hot, make_output, make_stats_group

"""
Load and preprocess data from the Heidelberg dataset.
"""

def load_shd(data, S_prime, digits, window_length, alphabet_size, pattern):

    samples = np.vstack([np.where(data.root.label == i) for i in digits])
    n_neurons = 700
    output = []

    res = []
    for i in samples:
        # variables to parse
        timestamps = data.root.spikes.times[i] * 10e6  # get times in mus
        addr = data.root.spikes.units[i]

        windows = list(range(window_length, min(int(timestamps[-1] + window_length), int((S_prime + 1) * window_length)), window_length))
        window_ptr = 0
        ts_pointer = 0

        current_group = []
        spiking_neurons_per_ts = [[] for _ in range(len(windows))]

        while (ts_pointer < len(timestamps)) & (window_ptr < len(windows)):
            if timestamps[ts_pointer] <= windows[window_ptr]:
                current_group += [addr[ts_pointer]]
            else:
                spiking_neurons_per_ts[window_ptr] += current_group
                window_ptr += 1
                current_group = [addr[ts_pointer]]
            ts_pointer += 1


        if alphabet_size == 1:
            if S_prime <= len(windows):
                input_signal = np.array([[1 if n in spiking_neurons_per_ts[s] else 0 for n in range(n_neurons)] for s in range(S_prime)])
                input_signal = input_signal.T[None, :, :]
            else:
                input_signal = np.array([[1 if n in spiking_neurons_per_ts[s] else 0 for n in range(n_neurons)] for s in range(len(windows))])
                padding = np.zeros([S_prime - len(windows), n_neurons])

                input_signal = np.vstack((input_signal, padding))
                input_signal = input_signal.T[None, :, :]


        else:
            if S_prime <= len(windows):
                input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, spiking_neurons_per_ts[s].count(n))) for n in range(n_neurons)] for s in range(S_prime)])
                input_signal = input_signal.transpose(1, 2, 0)[None, :]

            else:
                input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, spiking_neurons_per_ts[s].count(n))) for n in range(n_neurons)] for s in range(len(windows))])
                padding = np.zeros([S_prime - len(windows), n_neurons, alphabet_size])

                input_signal = np.vstack((input_signal, padding))
                input_signal = input_signal.transpose(1, 2, 0)[None, :]


        res.append(input_signal.astype(bool))
        output.append(make_output(data.root.labels[i], pattern, len(digits), alphabet_size, S_prime))

    return np.vstack(res).astype(bool), np.vstack(output).astype(bool)


def make_shd(path_to_train, path_to_test, path_to_hdf5, digits, alphabet_size, pattern, window_length):
    train_data_file = tables.open_file(path_to_train, 'r')
    test_data_file = tables.open_file(path_to_test, 'r')

    T_max = 1. * 1e6
    S_prime = math.ceil(T_max/window_length)

    hdf5_file = tables.open_file(path_to_hdf5, 'w')


    # Make train group and arrays
    train = hdf5_file.create_group(where=hdf5_file.root, name='train')

    train_data, output_signal = load_shd(path_to_train, S_prime, digits, window_length, alphabet_size, pattern)
    train_data_array = hdf5_file.create_array(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(), obj=train_data)
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), obj=output_signal)


    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_data, output_signal = load_shd(path_to_test, S_prime, digits, window_length, alphabet_size, pattern)
    test_data_array = hdf5_file.create_array(where=hdf5_file.root.test, name='data', atom=tables.BoolAtom(), obj=test_data)

    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='label', atom=tables.BoolAtom(), obj=output_signal)

    make_stats_group(hdf5_file)

    train_data_file.close()
    test_data_file.close()
    hdf5_file.close()


path_to_train = r'path/to/datasets/shd/shd_train.h5'
path_to_test = r'path/to/datasets/shd/shd_test.h5'

# digits to consider
digits = [i for i in range(20)]

# Pixel values to consider
window_length = 10000
alphabet_size = 2
pattern = [1]

if alphabet_size == 1:
        name = r'binary_%dms_%d_digits.hdf5' % (int(window_length / 1000), len(digits))
else:
    if alphabet_size == 2:
            name = r'%dms_%d_digits.hdf5' % (int(window_length / 1000), len(digits))
    else:
        name = r'_%dms_%d_digits_C_%d.hdf5' % (int(window_length / 1000), len(digits), alphabet_size)

path_to_hdf5 = r'path/to/datasets/shd/shd_' + name

make_shd(path_to_train, path_to_test, path_to_hdf5, digits, alphabet_size, pattern, window_length)
