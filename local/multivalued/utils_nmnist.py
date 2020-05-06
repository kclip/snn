import numpy as np
import os
import tables
import glob
from tqdm import tqdm

def one_hot(alphabet_size, idx):
    assert idx <= alphabet_size
    out = [0] * alphabet_size
    if idx > 0:
        out[idx - 1] = 1
    return out

def read_dataset(filename, sample_length=300000, window_length=10000):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7 #bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    n_neurons = 34**2

    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    #Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    addr = (all_x[td_indices] * 34 + all_y[td_indices]).tolist()
    all_ts = all_ts[td_indices].tolist()
    all_p = all_p[td_indices].tolist()

    windows = list(range(window_length, min(int(all_ts[-1] + window_length), int(sample_length + window_length)), window_length))
    window_ptr = 0
    ts_pointer = 0

    current_group = []
    spiking_neurons_per_ts = [[] for _ in range(len(windows))]

    while (ts_pointer < len(all_ts)) & (window_ptr < len(windows)):
        if all_ts[ts_pointer] <= windows[window_ptr]:
            current_group += [addr[ts_pointer] * (2 * all_p[ts_pointer] - 1)]
        else:
            spiking_neurons_per_ts[window_ptr] += current_group
            window_ptr += 1
            current_group = [addr[ts_pointer] * (2 * all_p[ts_pointer] - 1)]
        ts_pointer += 1

    S_prime = int(sample_length / window_length)

    if sample_length <= all_ts[-1]:
            input_signal = np.array([[[0, 1] if n in spiking_neurons_per_ts[s]
                                      else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0] for n in range(n_neurons)] for s in range(S_prime)])

            input_signal = input_signal.transpose(1, 2, 0)[None, :]

    else:
        input_signal = np.array([[[0, 1] if n in spiking_neurons_per_ts[s]
                                  else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0] for n in range(n_neurons)] for s in range(len(windows))])
        padding = np.zeros([S_prime - len(windows), n_neurons, 2])

        input_signal = np.vstack((input_signal, padding))
        input_signal = input_signal.transpose(1, 2, 0)[None, :]

    return input_signal.astype(bool)


def make_mnist_dvs(path_to_train, path_to_test, path_to_hdf5, digits, sample_length, window_length):

    """"
    Preprocess the .aedat file and save the dataset as an .hdf5 file
    """

    pattern = [1]   # the pattern used as output for the considered digit

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    S_prime = int(sample_length / window_length)

    train = hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_data_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(), shape=(0, 34**2, 2, S_prime))
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), shape=(0, len(digits), 2, S_prime))

    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_data_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='data', atom=tables.BoolAtom(), shape=(0, 34 **2, 2, S_prime))
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='label', atom=tables.BoolAtom(), shape=(0, len(digits), 2, S_prime))

    for i, digit in enumerate(digits):
        output_signal = np.vstack((np.array([[[0] * S_prime] * i
                                             + [pattern * int(S_prime / len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                             + [[0] * S_prime] * (len(digits) - 1 - i)], dtype=bool),
                                   np.zeros([1, len(digits), S_prime], dtype=bool))).transpose(1, 0, 2)[None, :]

        for dir_ in [r'/' + dir_ for dir_ in os.listdir(path_to_train)]:
                if dir_.find(str(digit)) != -1:
                    for subdir, _, _ in os.walk(path_to_train + dir_):
                        for j, file in enumerate(glob.glob(subdir + r'/*.bin')):
                            print('train', file)
                            train_data_array.append(read_dataset(file, sample_length=300000, window_length=10000))
                            train_labels_array.append(output_signal)


        for dir_ in [r'/' + dir_ for dir_ in os.listdir(path_to_test)]:
            if dir_.find(str(digit)) != -1:
                for subdir, _, _ in os.walk(path_to_test + dir_):
                    for j, file in enumerate(glob.glob(subdir + r'/*.bin')):
                        print('test', file)
                        test_data_array.append(read_dataset(file, sample_length=300000, window_length=10000))
                        test_labels_array.append(output_signal)

    train_shape = np.concatenate((hdf5_file.root.train.data[:], hdf5_file.root.train.label[:]), axis=1).shape
    test_shape = np.concatenate((hdf5_file.root.test.data[:], hdf5_file.root.test.label[:]), axis=1).shape

    stats = hdf5_file.create_group(where=hdf5_file.root, name='stats')
    train_data_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='train', atom=tables.Atom.from_dtype(np.dtype('int')),
                                                  obj=train_shape)
    test_data_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='test', atom=tables.Atom.from_dtype(np.dtype('int')),
                                                  obj=test_shape)

    hdf5_file.close()

digits = [i for i in range(10)]
window_length = 10000
path_to_hdf5 = r'C:/Users/K1804053/PycharmProjects/datasets/nmnist/nmnist_%dms_%d_digits.hdf5' % (int(window_length / 1000), len(digits))

make_mnist_dvs(r'C:\Users\K1804053\PycharmProjects\datasets\nmnist\Train', r'C:\Users\K1804053\PycharmProjects\datasets\nmnist\Test', path_to_hdf5, digits, 300000, window_length)