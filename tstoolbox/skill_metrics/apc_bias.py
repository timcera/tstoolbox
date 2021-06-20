# -*- coding: utf-8 -*-
import numpy as np

from . import utils


def apc_bias(simulated, observed):
    """
    Calculate the absolute percent bias between simulated and observed.

    B = 100.0*sum(abs(s-o))/sum(o)

    where s is the simulated values, and o is the observed values.
    Note that s & o must have the same number of values.

    Input:
    simulated : simulated field
    observed : observed field

    Output:
    B : absolute percent bias between simulated and observed
    """
    # Check that dimensions of simulated and observed fields match
    utils.check_arrays(simulated, observed)

    # Calculate bias in means
    b = 100.0 * np.sum(np.abs(simulated - observed)) / np.sum(observed)

    return b
