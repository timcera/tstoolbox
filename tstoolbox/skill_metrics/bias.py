# -*- coding: utf-8 -*-
import numpy as np

from . import utils


def bias(predicted, reference):
    """
    Calculate the bias between PREDICTED and REFERENCE.

    B = mean(p) - mean(r)

    where p is the predicted values, and r is the reference values.
    Note that p & r must have the same number of values.

    Input:
    PREDICTED : predicted field
    REFERENCE : reference field

    Output:
    B : bias between predicted and reference

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Dec 9, 2016
    """
    # Check that dimensions of predicted and reference fields match
    utils.check_arrays(predicted, reference)

    # Calculate bias in means
    b = np.mean(predicted) - np.mean(reference)

    return b
