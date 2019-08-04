from __future__ import absolute_import


from .classification import accuracy

from .retrieval import true_positive, true_negative, \
    false_positive, false_negative, \
    precision, recall, sensitivity, selectivity, fmeasure

from .similarity import jaccard_index, dice_coeff
