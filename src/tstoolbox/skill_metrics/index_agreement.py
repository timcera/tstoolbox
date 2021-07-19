# -*- coding: utf-8 -*-
import numpy as np

from . import utils


def index_agreement(simulated, observed):
    """
    Calculate the index of agreement.

    Calculates the index of agreement between two variables
    simulated and observed. The index_agreement is calculated using the
    formula:

    index_agreement = 1.0 - sum((o - s)**2) /
                            sum((abs(s - mean(o)) + abs(o - mean(o)))**2)

    where s is the simulated values, o is the observed values, and
    N is the total number of values in s & o. Note that s & o must
    have the same number of values.

    The index of agreement is between 0 and 1, where 1 is a perfect match.

    Input:
    simulated : simulated values
    observed : observed values

    Output:
    index_agreement : index of agreement
    """
    # Check that dimensions of simulated and observed fields match
    utils.check_arrays(simulated, observed)

    # Calculate the index_agreement
    index_agreement = 1.0 - (
        np.sum((observed - simulated) ** 2)
        / (
            np.sum(
                (
                    np.abs(simulated - np.mean(observed))
                    + np.abs(observed - np.mean(observed))
                )
                ** 2
            )
        )
    )

    return index_agreement
