#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""

from torch.nn.functional import relu, leaky_relu, max_pool2d, max_pool1d, dropout, dropout2d, upsample


def complex_relu(input_r, input_i, inplace=False):
    return relu(input_r, inplace), relu(input_i, inplace)


def complex_leaky_relu(input_r, input_i, negative_slope_r=0.01, negative_slope_i=0.01, inplace=False):
    return leaky_relu(input_r, negative_slope_r, inplace), leaky_relu(input_i, negative_slope_i, inplace)


def complex_max_pool2d(input_r, input_i, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):

    return max_pool2d(input_r, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices), \
        max_pool2d(input_i, kernel_size, stride, padding, dilation,
                   ceil_mode, return_indices)


def complex_max_pool1d(input_r, input_i, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):

    return max_pool1d(input_r, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices), \
        max_pool1d(input_i, kernel_size, stride, padding, dilation,
                   ceil_mode, return_indices)


def complex_dropout(input_r, input_i, p=0.5, training=True, inplace=False):
    return dropout(input_r, p, training, inplace), \
        dropout(input_i, p, training, inplace)


def complex_dropout2d(input_r, input_i, p=0.5, training=True, inplace=False):
    return dropout2d(input_r, p, training, inplace), \
        dropout2d(input_i, p, training, inplace)


def complex_upsample(input_r, input_i, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return upsample(input_r, size, scale_factor, mode, align_corners), \
        upsample(input_r, size, scale_factor, mode, align_corners)