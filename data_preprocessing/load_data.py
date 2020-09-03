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

import numpy as np
import scipy.misc
import tables
import h5py
import glob
from data_preprocessing.misc import *
import os
import torch


def get_example(hdf5_group, idx, T=80, n_classes=10, size=[1, 26, 26], dt=1000, polarity=True):
    data = np.empty([T] + size, dtype='float')
    label = hdf5_group.labels[idx, 0]
    start_time = hdf5_group.labels[idx, 1]
    end_time = hdf5_group.labels[idx, 2]

    curr = get_event_slice(times=hdf5_group.time[:], addrs=hdf5_group.data,
                           start_time=start_time, end_time=end_time, T=T, size=size, dt=dt, polarity=polarity)
    if len(curr) < T:
        data[:len(curr)] = curr
    else:
        data = curr[:T]

    return torch.FloatTensor(data.T), torch.FloatTensor(make_output_from_label(label, T, n_classes))


def get_event_slice(times, addrs, start_time, end_time, T, size=[128, 128], dt=1000, polarity=True):
    idx_beg = find_first(times, start_time)
    idx_end = find_first(times[idx_beg:], min(end_time, start_time + T * dt)) + idx_beg

    return chunk_evs_pol(times[idx_beg:idx_end], addrs[idx_beg:idx_end], deltat=dt, size=size, polarity=polarity)


def chunk_evs_pol(times, addrs, deltat=1000, size=[2, 304, 240], polarity=True):
    t_start = times[0]
    ts = range(t_start, times[-1], deltat)
    chunks = np.zeros([len(ts)]+size, dtype='int8')
    idx_start = 0
    idx_end = 0

    for i, t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t)
        if idx_end > idx_start:
            ee = addrs[idx_start:idx_end]
            if set(ee[:, 2]) == set([-1, 1]):  # Polarities are either [0, 1] or [-1, 1]
                pol, x, y = ((1 + ee[:, 2]) / 2).astype(np.int), ee[:, 0], ee[:, 1]
            else:
                pol, x, y = ee[:, 2], ee[:, 0], ee[:, 1]
            try:
                if len(size) == 3:
                    np.add.at(chunks, (i, pol, x, y), 1)
                elif len(size) == 2:
                    np.add.at(chunks, (i, pol, (x * 32 + y).astype(int)), 1)
                elif len(size) == 1:
                    if polarity:
                        chunks[i, (pol + 2 * (x * 32 + y)).astype(int)] = 1
                    else:
                        chunks[i, (x * 32 + y).astype(int)] = 1
            except:
                i_max = np.max((pol + 2 * (x * 32 + y)))
                print(x[i_max], y[i_max], pol[i_max])
                raise IndexError
        idx_start = idx_end
    return chunks
