import time
import os
import fnmatch
import numpy as np


def mksavedir(pre='results/', exp_dir=None):
    """
    Creates a results directory in the subdirectory 'pre'
    """

    if pre[-1] != '/':
        pre + '/'

    if not os.path.exists(pre):
        os.makedirs(pre)
    prelist = np.sort(fnmatch.filter(os.listdir(pre), '[0-9][0-9][0-9]__*'))

    if exp_dir is None:
        if len(prelist) == 0:
            expDirN = "001"
        else:
            expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

        save_dir = time.strftime(pre + expDirN + "__" + "%d-%m-%Y", time.localtime())

    elif isinstance(exp_dir, str):
        if len(prelist) == 0:
            expDirN = "001"
        else:
            expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

        save_dir = time.strftime(pre + expDirN + "__" + "%d-%m-%Y", time.localtime()) + '_' + exp_dir

    else:
        raise TypeError('exp_dir should be a string')


    os.makedirs(save_dir)
    print(("Created experiment directory {0}".format(save_dir)))
    return save_dir + r'/'
