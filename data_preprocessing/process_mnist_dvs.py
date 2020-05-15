import tables
import os
import glob
import numpy as np
import math
import struct
from data_preprocessing.misc import one_hot, make_output, make_stats_group

"""
Load and preprocess data from the MNIST-DVS dataset. The data samples had previously been processed using matlab scripts provided by the authors. 
See http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html
"""


def load_dvs(datafile, S_prime, min_pxl_value=48, max_pxl_value=80, window_length=25000, alphabet_size=None, polarity=True):
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    xmask = 0x00fe
    ymask = 0x7f00
    pmask = 0x1

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)

    length = statinfo.st_size

    # header
    lt = aerdatafh.readline()
    while lt and lt[:1] == b'#':
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    polarities = []

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    while p < length:
        addr, ts = struct.unpack(readMode, s)

        # parse event's data
        x_addr = 128 - 1 - ((xmask & addr) >> 1)
        y_addr = ((ymask & addr) >> 8)
        a_pol = 1 - 2 * (addr & pmask)

        if (x_addr >= min_pxl_value) & (x_addr <= max_pxl_value) & (y_addr >= min_pxl_value) & (y_addr <= max_pxl_value):
            timestamps.append(ts)
            xaddr.append(x_addr - min_pxl_value)
            yaddr.append(y_addr - min_pxl_value)
            polarities.append(a_pol)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    # create windows to store eventa
    windows = list(range(window_length - 1000, max(timestamps), window_length))
    window_ptr = 0
    ts_pointer = 0
    n_neurons = (max_pxl_value - min_pxl_value + 1)**2

    timestamps_grouped = [[] for _ in range(len(windows))]
    current_group = []

    while (ts_pointer < len(timestamps)) & (window_ptr < len(windows)):
        if timestamps[ts_pointer] <= windows[window_ptr]:
            current_group += [ts_pointer]
        else:
            timestamps_grouped[window_ptr] += current_group
            window_ptr += 1
            current_group = [ts_pointer]
        ts_pointer += 1

    if alphabet_size == 1:
        if polarity:
            spiking_neurons_per_ts = [[(xaddr[ts] * (max_pxl_value - min_pxl_value + 1) + yaddr[ts]) * polarities[ts] for ts in group] for group in timestamps_grouped]
            if S_prime <= len(windows):
                input_signal = np.array(
                    [[[0, 1] if n in spiking_neurons_per_ts[s] else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0] for n in range(n_neurons)] for s in range(S_prime)])
                input_signal = input_signal.reshape([S_prime, -1]).T[None, :]
            else:
                input_signal = np.array(
                    [[[0, 1] if n in spiking_neurons_per_ts[s] else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0] for n in range(n_neurons)] for s in range(len(windows))])
                padding = np.zeros([S_prime - len(windows), n_neurons, 2])
                input_signal = np.vstack((input_signal, padding))
                input_signal = input_signal.reshape([S_prime, -1]).T[None, :]
        else:
            spiking_neurons_per_ts = [[xaddr[ts] * (max_pxl_value - min_pxl_value + 1) + yaddr[ts] for ts in group] for group in timestamps_grouped]
            if S_prime <= len(windows):
                input_signal = np.array([[1 if n in spiking_neurons_per_ts[s] else 0 for n in range(n_neurons)] for s in range(S_prime)])
                input_signal = input_signal.T[None, :, :]
            else:
                input_signal = np.array([[1 if n in spiking_neurons_per_ts[s] else 0 for n in range(n_neurons)] for s in range(len(windows))])
                padding = np.zeros([S_prime - len(windows), n_neurons])

                input_signal = np.vstack((input_signal, padding))
                input_signal = input_signal.T[None, :, :]

    else:
        if polarity:
            spiking_neurons_per_ts = [[(xaddr[ts] * (max_pxl_value - min_pxl_value + 1) + yaddr[ts]) * polarities[ts] for ts in group] for group in timestamps_grouped]
            if S_prime <= len(windows):
                input_signal = np.array([[[0, 1] if n in spiking_neurons_per_ts[s] else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0]
                                          for n in range(n_neurons)] for s in range(S_prime)])
                input_signal = input_signal.transpose(1, 2, 0)[None, :]

            else:
                input_signal = np.array([[[0, 1] if n in spiking_neurons_per_ts[s] else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0]
                                          for n in range(n_neurons)] for s in range(len(windows))])
                padding = np.zeros([S_prime - len(windows), n_neurons, 2])

                input_signal = np.vstack((input_signal, padding))
                input_signal = input_signal.transpose(1, 2, 0)[None, :]

        else:
            spiking_neurons_per_ts = [[xaddr[ts] * (max_pxl_value - min_pxl_value + 1) + yaddr[ts] for ts in group] for group in timestamps_grouped]

            if S_prime <= len(windows):
                input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, max(0, spiking_neurons_per_ts[s].count(n) + spiking_neurons_per_ts[s].count(-n))))
                                          for n in range(n_neurons)] for s in range(S_prime)])
                input_signal = input_signal.transpose(1, 2, 0)[None, :]

            else:
                input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, spiking_neurons_per_ts[s].count(n) + spiking_neurons_per_ts[s].count(-n)))
                                          for n in range(n_neurons)] for s in range(len(windows))])
                padding = np.zeros([S_prime - len(windows), n_neurons, alphabet_size])

                input_signal = np.vstack((input_signal, padding))
                input_signal = input_signal.transpose(1, 2, 0)[None, :]

    return input_signal.astype(bool)




def make_mnist_dvs(path_to_data, path_to_hdf5, digits, max_pxl_value, min_pxl_value, T_max, window_length, scale, polarity, pattern, alphabet_size):
    """"
    Preprocess the .aedat file and save the dataset as an .hdf5 file
    """

    dirs = [r'/' + dir_ for dir_ in os.listdir(path_to_data)]

    S_prime = math.ceil(T_max/window_length)

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    train = hdf5_file.create_group(where=hdf5_file.root, name='train')

    if alphabet_size == 1:
        data_shape = (0, (1 + polarity)*(max_pxl_value - min_pxl_value + 1)**2, S_prime)
        label_shape = (0, len(digits), S_prime)
    else:
        data_shape = (0, (max_pxl_value - min_pxl_value + 1)**2, alphabet_size, S_prime)
        label_shape = (0, len(digits), alphabet_size, S_prime)

    train_data = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(), shape=data_shape)
    train_labels = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), shape=label_shape)

    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_data = hdf5_file.create_earray(where=hdf5_file.root.test, name='data', atom=tables.BoolAtom(), shape=data_shape)
    test_labels = hdf5_file.create_earray(where=hdf5_file.root.test, name='label', atom=tables.BoolAtom(), shape=label_shape)


    for i, digit in enumerate(digits):
        output_signal = make_output(i, pattern, len(digits), alphabet_size, S_prime)

        for dir_ in dirs:
                if dir_.find(str(digit)) != -1:
                    for subdir, _, _ in os.walk(path_to_data + dir_):
                        if subdir.find(scale) != -1:
                            for j, file in enumerate(glob.glob(subdir + r'/*.aedat')):
                                if j < 0.9*len(glob.glob(subdir + r'/*.aedat')):
                                    print('train', file)
                                    tmp = load_dvs(file, S_prime, min_pxl_value=min_pxl_value, max_pxl_value=max_pxl_value, window_length=window_length, polarity=polarity)
                                    print(tmp.shape, data_shape)

                                    train_data.append(tmp)
                                    train_labels.append(output_signal)
                                else:
                                    print('test', file)
                                    test_data.append(load_dvs(file, S_prime, min_pxl_value=min_pxl_value, max_pxl_value=max_pxl_value,
                                                              window_length=window_length, polarity=polarity))

                                    test_labels.append(output_signal)

    make_stats_group(hdf5_file)

    hdf5_file.close()


path_to_data = r'path/to/mnist-dvs/processed_polarity'

# digits to consider
digits = [1, 7]

# Pixel values to consider
max_pxl_value = 73
min_pxl_value = 48

T_max = int(2e6)  # maximum duration of an example in us
window_length = 10000

scale = 'scale4'
polarity = True
alphabet_size = 2
pattern = [1]

if alphabet_size == 1:
    if polarity:
        name = r'binary_%dms_%dpxl_%d_digits_polarity.hdf5' % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits))
    else:
        name = r'binary_%dms_%dpxl_%d_digits.hdf5' % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits))
else:
    if alphabet_size == 2:
        if polarity:
            name = r'%dms_%dpxl_%d_digits_polarity.hdf5' % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits))
        else:
            name = r'%dms_%dpxl_%d_digits.hdf5' % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits))
    else:
        name = r'_%dms_%dpxl_%d_digits_C_%d.hdf5' % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits), alphabet_size)
path_to_hdf5 = r'path/to/datasets/mnist-dvs/mnist_dvs_' + name

make_mnist_dvs(path_to_data, path_to_hdf5, digits, max_pxl_value, min_pxl_value, T_max, window_length, scale, polarity, pattern, alphabet_size)





