import bisect

import numpy as np
import tables
import torch

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


def make_output_from_labels(labels, T, classes, size):
    if len(size) == 1:
        return make_outputs_binary(labels, T, classes)
    elif len(size) == 2:
        return make_outputs_multivalued(labels, T, classes)
    else:
        raise NotImplementedError


def make_outputs_binary(labels, T, classes):
    mapping = {classes[i]: i for i in range(len(classes))}

    if hasattr(labels, 'len'):
        out = torch.zeros([len(labels), len(classes), T])
        out[[i for i in range(len(labels))], [mapping[lbl] for lbl in labels], :] = 1
    else:
        out = torch.zeros([len(classes), T])
        mapping = {classes[i]: i for i in range(len(classes))}
        out[mapping[labels], :] = 1
    return out


def make_outputs_multivalued(labels, T, classes):
    mapping = {classes[i]: i for i in range(len(classes))}

    if hasattr(labels, 'len'):
        out = torch.zeros([len(labels), len(classes), 2, T])
        out[[i for i in range(len(labels))], [mapping[lbl] for lbl in labels], 0, :] = 1
    else:
        out = torch.zeros([len(classes), 2, T])
        mapping = {classes[i]: i for i in range(len(classes))}
        out[mapping[labels], 0, :] = 1
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
