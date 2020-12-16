#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


from __future__ import division, print_function, absolute_import
import h5py
import numpy as np
import scipy.io as scio


def loadmat(filename):

    return scio.loadmat(filename)


def savemat(filename, mdict, fmt='5', dtype=None):
    for k, v in mdict.items():
        if np.iscomplex(v).any() and np.ndim(v) > 1:
            mdict[k] = np.array(
                [np.real(v), np.imag(v)]).transpose(1, 2, 0)
            mdict[k] = mdict[k].astype('float32')
    scio.savemat(filename, mdict, format=fmt)

    return 0


def _create_group_dataset(group, mdict):
    for k, v in mdict.items():
        if type(v) is dict:
            subgroup = group.create_group(k)
            _create_group_dataset(subgroup, v)
        else:
            group.create_dataset(k, data=v)


def _read_group_dataset(group, mdict):
    for k in group.keys():
        if type(group[k]) is h5py.Group:
            exec(k + '={}')
            _read_group_dataset(group[k], eval(k))
            mdict[k] = eval(k)
        else:
            mdict[k] = group[k][()]


def loadh5(filename):
    """load h5 file

    load all the data from a h5 file.

    Parameters
    ----------
    filename : {str}
        File's full path string.

    Returns
    -------
    D : {dict}
        The loaded data in ``dict`` type.

    """

    f = h5py.File(filename, 'r')
    D = {}

    _read_group_dataset(f, D)

    f.close()
    return D


def saveh5(filename, mdict):
    """save data to h5 file

    save data to h5 file

    Parameters
    ----------
    filename : {str}
        filename string
    mdict : {dict}
        each dict is store in group, the elements in dict are store in dataset

    Returns
    -------
    number
        0 --> all is well.
    """

    f = h5py.File(filename, 'w')

    _create_group_dataset(f, mdict)

    f.close()
    return 0


if __name__ == '__main__':

    a = np.random.randn(3, 4)
    b = 10
    c = [1, 2, 3]
    d = {'1': 1, '2': a}
    s = 'Hello, the future!'

    saveh5('./data.h5', {'a': a, 'b': b, 'c': c, 'd': d, 's': s})

    data = loadh5('./data.h5')

    print(data)
