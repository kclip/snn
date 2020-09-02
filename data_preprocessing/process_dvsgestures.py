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


def aedat_to_events(filename):
    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename, skiprows=1, delimiter=',', dtype='uint32')
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

                x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])

            else:
                f.read(eventnumber * eventsize)
    events = np.column_stack(events)
    events = events.astype('uint32')
    clipped_events = np.zeros([4, 0], 'uint32')

    for l in labels:
        start = np.searchsorted(events[0, :], l[1])
        end = np.searchsorted(events[0, :], l[2])
        clipped_events = np.column_stack([clipped_events, events[:, start:end]])

    return clipped_events.T, labels


def create_events_hdf5(path_to_hdf5, path_to_data):
    fns_train = gather_aedat(path_to_data, 1, 24)
    fns_test = gather_aedat(path_to_data, 24, 30)

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_times_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='time', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0,))
    train_data_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='labels', atom=tables.Atom.from_dtype(np.dtype('int64')), shape=(0, 3))


    print("processing training data...")
    key = 0

    for file_d in fns_train:
        print(key)
        events, labels = aedat_to_events(file_d)
        subgrp = train_grp.create_group(str(key))
        dset_dt = subgrp.create_dataset('time', events[:, 0].shape, dtype=np.uint32)
        dset_da = subgrp.create_dataset('data', events[:, 1:].shape, dtype=np.uint8)
        dset_dt[...] = events[:, 0]
        dset_da[...] = events[:, 1:]
        dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
        dset_l[...] = labels
        key += 1

        print("processing testing data...")
        key = 0
        test_grp = f.create_group('test')
        for file_d in fns_test:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = test_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:, 0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:, 1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:, 0]
            dset_da[...] = events[:, 1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1

        stats = gather_gestures_stats(train_grp)
        f.create_dataset('stats', stats.shape, dtype=stats.dtype)
        f['stats'][:] = stats

        stats_train_data = np.array([9000, (1 + max_pxl_value - min_pxl_value)])
        stats_train_label = np.array([9000, 10])

        stats_test_data = np.array([1000, (1 + max_pxl_value - min_pxl_value)])
        stats_test_label = np.array([1000, 10])

        hdf5_file.create_group(where=hdf5_file.root, name='stats')
        hdf5_file.create_array(where=hdf5_file.root.stats, name='train_data', atom=tables.Atom.from_dtype(stats_train_data.dtype), obj=stats_train_data)
        hdf5_file.create_array(where=hdf5_file.root.stats, name='train_label', atom=tables.Atom.from_dtype(stats_train_label.dtype), obj=stats_train_label)
        hdf5_file.create_array(where=hdf5_file.root.stats, name='test_data', atom=tables.Atom.from_dtype(stats_test_data.dtype), obj=stats_test_data)
        hdf5_file.create_array(where=hdf5_file.root.stats, name='test_label', atom=tables.Atom.from_dtype(stats_test_label.dtype), obj=stats_test_label)


def create_data(filename=os.path.join(dcll_folder, '../data/dvs_gestures_events.hdf5')):
    if not os.path.isfile(filename):
        print("File {} does not exist: converting DvsGesture to h5file".format(filename))
        create_events_hdf5(filename)
    else:
        print("File {} exists: not re-converting DvsGesture".format(filename))
