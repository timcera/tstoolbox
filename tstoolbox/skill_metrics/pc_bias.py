import numpy as np

from . import utils


def pc_bias(simulated, observed):
    """
    Calculate the percent bias between simulated and observed.

    B = 100.0*sum(s-o)/sum(o)

    where s is the simulated values, and o is the observed values.
    Note that s & o must have the same number of values.

    Input:
    simulated : simulated field
    observed : observed field

    Output:
    B : percent bias between simulated and observed
    """
    # Check that dimensions of simulated and observed fields match
    utils.check_arrays(simulated, observed)

    # Calculate bias in means
    b = 100.0 * np.sum(simulated - observed) / np.sum(observed)

    return b
