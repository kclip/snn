import tables
import os
import glob
from utils_dvs import load_dvs
from utils_heidelberg import load_shd_accum
import re
import numpy as np
import math
import struct


def process_dvs(datafile, output_file, current_time, T_max, min_pxl_value, max_pxl_value):
    # Pixel values to consider

    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us

    xmask = 0x00fe
    ymask = 0x7f00

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

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    ts_last = 0.0

    with open(output_file, 'w') as f:
        while p < length:
            addr, ts = struct.unpack(readMode, s)

            # parse event's data
            x_addr = 128 - 1 - ((xmask & addr) >> 1)
            y_addr = ((ymask & addr) >> 8)

            if (x_addr >= min_pxl_value) & (x_addr <= max_pxl_value) & (y_addr >= min_pxl_value) & (y_addr <= max_pxl_value) & (ts < T_max):
                f.write("%2f %d\n" % (current_time / 100, (x_addr - min_pxl_value) * (max_pxl_value - min_pxl_value + 1) + (y_addr - min_pxl_value)))

                if int(ts * 1e-4) != int(ts_last * 1e-4):
                    current_time += int(ts * 1e-4) - int(ts_last * 1e-4)
                    ts_last = ts

            aerdatafh.seek(p)
            s = aerdatafh.read(aeLen)
            p += aeLen

        current_time += 10

    return current_time


def make_output(output_file, T_max, current_time, digit):
    with open(output_file, 'w') as f:
        for i in range(current_time, current_time + int(T_max * 1e-4)):
            if (i % 10) == 0:
                f.write("%2f %d\n" % (i / 100, digit))


def make_mnist_dvs(path_to_data, path_to_ras, digits, max_pxl_value, min_pxl_value, T_max, scale):

    """"
    Preprocess the .aedat file and save the dataset as an .hdf5 file
    """

    dirs = [r'/' + dir_ for dir_ in os.listdir(path_to_data)]

    input_train = path_to_ras + '/mnist_dvs_%d_digits_input_train.ras'
    output_train = path_to_ras + '/mnist_dvs_%d_digits_output_train.ras'

    input_test = path_to_ras + '/mnist_dvs_%d_digits_input_test.ras'
    output_test = path_to_ras + '/mnist_dvs_%d_digits_output_test.ras'

    current_time = 0

    for i, digit in enumerate(digits):
        for dir_ in dirs:
                if dir_.find(str(digit)) != -1:
                    for subdir, _, _ in os.walk(path_to_data + dir_):
                        if subdir.find(scale) != -1:
                            for j, file in enumerate(glob.glob(subdir + r'/*.aedat')):
                                if j < 0.9*len(glob.glob(subdir + r'/*.aedat')):
                                    print('train', file)
                                    current_time = process_dvs(file, input_train, current_time, T_max, min_pxl_value, max_pxl_value)
                                    make_output(output_train, T_max, current_time, i)

                                else:
                                    print('test', file)
                                    current_time = process_dvs(file, input_test, current_time, T_max, min_pxl_value, max_pxl_value)
                                    make_output(output_test, T_max, current_time, i)



if __name__ == "__main__":
    path_to_data = r'C:\Users\K1804053\Desktop\PhD\Federated SNN\processed'

    # digits to consider
    digits = [i for i in range(10)]

    # Pixel values to consider
    max_pxl_value = 73
    min_pxl_value = 48

    T_max = int(2e6)  # maximum duration of an example in us

    scale = 'scale4'

    path_to_ras = r'C:/Users/K1804053/PycharmProjects/datasets/mnist-dvs'

    make_mnist_dvs(path_to_data, path_to_ras, digits, max_pxl_value, min_pxl_value, T_max, scale)
