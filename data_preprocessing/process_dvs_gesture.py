import numpy as np
import os
import struct
import tables
from tqdm import tqdm
from data_preprocessing.misc import make_stats_group
import glob


def gather_aedat(directory, start_id, end_id, filename_prefix='user'):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found, looked at: {}".format(directory))

    fns = []

    for i in range(start_id, end_id):
        search_mask = directory + '/' + filename_prefix + "{0:02d}".format(i) + '*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out) > 0:
            fns += glob_out

    return fns


def load_dvs(datafile, S_prime, classes, alphabet_size, pattern, window_length=5000, sample_length=500000, grid_size=128, reduction_factor=4):
    label_filename = datafile[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename, skiprows=1, delimiter=',', dtype='uint32')

    # variables to parse
    events = []

    pxl_groups = []
    for i in range(0, grid_size, reduction_factor):
        for j in range(0, grid_size, reduction_factor):
            pxl_groups.append([i_ * grid_size + j_ for i_ in range(i, i + reduction_factor) for j_ in range(j, j + reduction_factor)])

    ts_current = 0

    with open(datafile, 'rb') as f:
        for i in range(5):
            f.readline()

        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0:
                break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]

            if eventtype == 1:
                event_bytes = np.frombuffer(f.read(eventnumber*eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                xaddr = (event_bytes[:, 0] >> 17) & 0x00001FFF
                yaddr = (event_bytes[:, 0] >> 2) & 0x00001FFF

                polarity = (event_bytes[:, 0] >> 1) & 0x00000001
                timestamp = event_bytes[:, 1].copy()

                for i, ts in enumerate(timestamp):
                    if (ts - ts_current) > window_length:
                        ts_current += window_length*int((ts - ts_current)/window_length)
                    timestamp[i] = ts_current

                addr = (xaddr // reduction_factor) * (grid_size // reduction_factor) + (yaddr // reduction_factor)

                events.append([timestamp, addr, polarity])
            else:
                f.read(eventnumber*eventsize)

    events = np.column_stack(events)

    events[0, :] -= events[0, 0] # make everything start at 0s
    events = events.astype('uint32')
    clipped_events = []

    labels[:, 1:] -= labels[0, 1]  # make everything start at 0s

    for l in labels:
        if l[0] in classes:
            start = np.searchsorted(events[0, :], l[1])
            end = np.searchsorted(events[0, :], l[2])

            clipped_events.append(events[:, start:end])

    input_signal = np.zeros([0, int(grid_size / reduction_factor) ** 2, alphabet_size, S_prime])
    output_signal = np.zeros([0, len(classes), alphabet_size, S_prime])

    for i in range(len(classes)):
        clipped_events[i][0, :] -= clipped_events[i][0, 0]  # make everything start at 0 for the event

        num_samples = int(clipped_events[i][0, -1] / sample_length)
        current_input = np.zeros([num_samples, int(grid_size / reduction_factor) ** 2, alphabet_size, S_prime])

        output = np.stack((np.array([[[0] * S_prime] * i + [pattern * int(S_prime / len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                     + [[0] * S_prime] * (len(classes) - 1 - i)], dtype=bool),
                           np.zeros([1, len(classes), S_prime])), axis=-2)
        for j in range(num_samples):
            output_signal = np.vstack((output_signal, output))

            for l, ts in enumerate(range(j * sample_length, (j + 1) * sample_length, window_length)):
                event_idx = np.where(clipped_events[i][0, :] == ts)[0]
                if len(event_idx) > 0:
                    spikes = np.array([[1, 0] if (clipped_events[i][2, event_idx[k]] == 1) else [0, 1] for k in range(len(event_idx))])
                    current_input[j, clipped_events[i][1, event_idx], :, l] = spikes


        input_signal = np.vstack((input_signal, current_input))

    return input_signal, output_signal


def create_events_hdf5(directory, path_to_hdf5, classes, alphabet_size, pattern, grid_size=128, reduction_factor=4, sample_length_train=500000, sample_length_test=1800000, window_length=5000):
    fns_train = gather_aedat(directory, 1, 24)
    fns_test = gather_aedat(directory, 24, 30)

    print(len(fns_train), len(fns_test))
    assert len(fns_train) == 98

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    n_neurons = int(grid_size / reduction_factor) ** 2

    S_prime_train = int(np.ceil(sample_length_train / window_length))
    S_prime_test = int(np.ceil(sample_length_test / window_length))
    
    if alphabet_size == 1:
        data_shape_train = (0, n_neurons, S_prime_train)
        label_shape_train = (0, len(classes), S_prime_train)
        
        data_shape_test = (0, n_neurons, S_prime_test)
        label_shape_test = (0, len(classes), S_prime_test)

    else:
        data_shape_train = (0, n_neurons, alphabet_size, S_prime_train)
        label_shape_train = (0, len(classes), alphabet_size, S_prime_train)
        
        data_shape_test = (0, n_neurons, alphabet_size, S_prime_test)
        label_shape_test = (0, len(classes), alphabet_size, S_prime_test)


    train = hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_data_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(), shape=data_shape_train)
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), shape=label_shape_train)

    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_data_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='data', atom=tables.BoolAtom(), shape=data_shape_test)
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='label', atom=tables.BoolAtom(), shape=label_shape_test)

    for file_d in tqdm(fns_train+fns_test):
        istrain = file_d in fns_train
        if istrain:
            input, output = load_dvs(file_d, S_prime_train, classes, alphabet_size, pattern, window_length, sample_length_train, grid_size, reduction_factor)
            train_data_array.append(input)
            train_labels_array.append(output)
        else:
            input, output = load_dvs(file_d, S_prime_test, classes, alphabet_size, pattern, window_length, sample_length_test, grid_size, reduction_factor)
            test_data_array.append(input)
            test_labels_array.append(output)

    make_stats_group(hdf5_file)

    hdf5_file.close()


classes = [i + 1 for i in range(11)]
alphabet_size = 2
pattern = [1]
sample_length_train = 500000
sample_length_test = 1800000
window_length = 15000
grid_size = 128
reduction_factor = 4

if alphabet_size == 1:
        name = r'binary_%dms_%d_digits.hdf5' % (int(window_length / 1000), len(classes))
else:
    if alphabet_size == 2:
            name = r'%dms_%d_digits.hdf5' % (int(window_length / 1000), len(classes))
    else:
        name = r'_%dms_%d_digits_C_%d.hdf5' % (int(window_length / 1000), len(classes), alphabet_size)

path_to_hdf5 = r'path/to/datasets/DvsGesture/dvs_gesture_' + name


create_events_hdf5(r'/path/to/DvsGesture', path_to_hdf5, classes, alphabet_size, pattern,
                   grid_size, reduction_factor, sample_length_train, sample_length_test, window_length)

