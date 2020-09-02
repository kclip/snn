import numpy as np
import tables
import bisect


def make_output(label, pattern, num_labels, alphabet_size, S_prime):
    if alphabet_size ==1:
        output_signal = np.array([[[0] * S_prime] * label
                                  + [pattern * int(S_prime / len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                  + [[0] * S_prime] * (num_labels - 1 - label)], dtype=bool)
    else:
        output_signal = np.vstack((np.array([[[0] * S_prime] * label
                                             + [pattern * int(S_prime / len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                             + [[0] * S_prime] * (num_labels - 1 - label)], dtype=bool),
                                   np.zeros([alphabet_size - 1, num_labels, S_prime], dtype=bool))).transpose(1, 0, 2)[None, :]

    return output_signal


def one_hot(alphabet_size, idx):
    assert idx <= alphabet_size
    out = [0]*alphabet_size
    if idx > 0:
        out[idx - 1] = 1
    return out


def expand_targets(targets, T=500, burnin=0):
    y = np.tile(targets.copy(), [T, 1, 1])
    y[:burnin] = 0
    return y


def make_output_from_label(label, T, num_classes):
    out = np.zeros([num_classes, T])
    out[label, :] = 1
    return out


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)


def make_stats_group(hdf5_file):
    train_data_shape = hdf5_file.root.train.data[:].shape
    train_label_shape = hdf5_file.root.train.label[:].shape
    test_data_shape = hdf5_file.root.test.data[:].shape
    test_label_shape = hdf5_file.root.test.label[:].shape


    stats = hdf5_file.create_group(where=hdf5_file.root, name='stats')
    train_data_stats_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='train_data', atom=tables.Atom.from_dtype(np.dtype('int')), obj=train_data_shape)
    train_label_stats_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='train_label', atom=tables.Atom.from_dtype(np.dtype('int')), obj=train_label_shape)
    test_data_stats_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='test_data', atom=tables.Atom.from_dtype(np.dtype('int')), obj=test_data_shape)
    test_label_stats_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='test_label', atom=tables.Atom.from_dtype(np.dtype('int')), obj=test_label_shape)
