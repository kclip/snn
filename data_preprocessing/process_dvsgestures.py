#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : Fri 01 Dec 2017 10:05:17 PM PST
# Last Modified : Sun 29 Jul 2018 01:39:06 PM PDT
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
# -----------------------------------------------------------------------------
import struct
import numpy as np
import scipy.misc
import tables
import glob
from misc import *
import os


mapping = {0: 'Hand Clapping',
           1: 'Right Hand Wave',
           2: 'Left Hand Wave',
           3: 'Right Arm CW',
           4: 'Right Arm CCW',
           5: 'Left Arm CW',
           6: 'Left Arm CCW',
           7: 'Arm Roll',
           8: 'Air Drums',
           9: 'Air Guitar',
           10: 'Other'}



def gather_gestures_stats(hdf5_grp):
    from collections import Counter
    labels = []
    for d in hdf5_grp:
        labels += hdf5_grp[d]['labels'][:, 0].tolist()
    count = Counter(labels)
    stats = np.array(list(count.values()))
    stats = stats / stats.sum()
    return stats


def gather_aedat(directory, start_id, end_id, filename_prefix='user'):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found, looked at: {}".format(directory))
    import glob
    fns = []
    for i in range(start_id, end_id):
        search_mask = directory + '/' + filename_prefix + "{0:02d}".format(i) + '*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out) > 0:
            fns += glob_out
    return fns


def aedat_to_events(filename, last_ts=0, ds=1):
    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename, skiprows=1, delimiter=',', dtype='int64')
    events = []
    with open(filename, 'rb') as f:
        for i in range(5):
            f.readline()
        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0: break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if (eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = ((event_bytes[:, 0] >> 17) & 0x00001FFF) // ds
                y = ((event_bytes[:, 0] >> 2) & 0x00001FFF) // ds
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])

            else:
                f.read(eventnumber * eventsize)
    events = np.column_stack(events)
    events = events.astype('int64')
    clipped_events = np.zeros([4, 0], 'int64')

    for l in labels:
        start = np.searchsorted(events[0, :], l[1])
        end = np.searchsorted(events[0, :], l[2])
        clipped_events = np.column_stack([clipped_events, events[:, start:end]])

    # Normalize times
    clipped_events[0, :] -= labels[0, 1]
    labels[:, 1:] -= labels[0, 1]
    clipped_events[0, :] += last_ts + 10000
    labels[:, 1:] += last_ts + 10000
    labels[:, 0] -= 1

    return clipped_events, labels


def create_events_hdf5(path_to_hdf5, path_to_data, ds=1):
    fns_train = gather_aedat(path_to_data, 1, 24)
    fns_test = gather_aedat(path_to_data, 24, 30)

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_times_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='time', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0,))
    train_data_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='labels', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))


    print("processing training data...")
    last_ts = 0
    for file_d in fns_train:
        events, labels = aedat_to_events(file_d, last_ts, ds)
        train_labels_array.append(labels)
        train_times_array.append(events[0, :])
        train_data_array.append(events[1:, :].T)

        last_ts = events[0, -1]



    hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_times_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='time', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0,))
    test_data_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='data', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='labels', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))

    print("processing testing data...")
    last_ts = 0
    for file_d in fns_test:
        events, labels = aedat_to_events(file_d, last_ts, ds)
        test_labels_array.append(labels)
        test_times_array.append(events[0, :])
        test_data_array.append(events[1:, :].T)

        last_ts = events[0, -1]

    stats_train_data = np.array([len(hdf5_file.root.train.labels[:]), 32])
    stats_train_label = np.array([len(hdf5_file.root.train.labels[:]), 11])

    stats_test_data = np.array([len(hdf5_file.root.test.labels[:]), 32])
    stats_test_label = np.array([len(hdf5_file.root.test.labels[:]), 11])

    hdf5_file.create_group(where=hdf5_file.root, name='stats')
    hdf5_file.create_array(where=hdf5_file.root.stats, name='train_data', atom=tables.Atom.from_dtype(stats_train_data.dtype), obj=stats_train_data)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='train_label', atom=tables.Atom.from_dtype(stats_train_label.dtype), obj=stats_train_label)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='test_data', atom=tables.Atom.from_dtype(stats_test_data.dtype), obj=stats_test_data)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='test_label', atom=tables.Atom.from_dtype(stats_test_label.dtype), obj=stats_test_label)


def create_data(path_to_hdf5='../data/mnist_dvs_events.hdf5', path_to_data=None, ds=1):
    if os.path.exists(path_to_hdf5):
        print("File {} exists: not re-converting data".format(path_to_hdf5))
    elif (not os.path.exists(path_to_hdf5)) & (path_to_data is not None):
        print("converting DvsGestures to h5file")
        create_events_hdf5(path_to_hdf5, path_to_data, ds)
    else:
        print('Either an hdf5 file or DvsGestures data must be specified')


# create_data(path_to_hdf5=r'C:\Users\K1804053\PycharmProjects\datasets\DvsGesture\dvs_gestures_events.hdf5',
#             path_to_data=r'C:\Users\K1804053\PycharmProjects\datasets\DvsGesture',
#             ds=4)

create_data(path_to_hdf5=r'/users/k1804053/datasets/DvsGesture/dvs_gestures_events.hdf5',
            path_to_data=r'/users/k1804053/DvsGesture',
            ds=4)
